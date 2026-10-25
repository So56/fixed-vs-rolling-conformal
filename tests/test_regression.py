"""Step A baseline tests. Run with unittest; inputs are immutable saved snapshots.

No analysis module is imported. Floating comparisons are exact, with NaN positions retained.
The small invariants below also reject deliberately corrupted in-memory copies.
"""
from pathlib import Path
import hashlib,json,math,os,unittest
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[2]
VALIDATION=Path(os.environ.get('STEP_A_VALIDATION',PROJECT/'refactor_validation/step_a'))
SAVED=VALIDATION/'baseline_saved'
def read(path):return pd.read_csv(SAVED/path,float_precision='round_trip')
def js(path):return json.loads((SAVED/path).read_text(encoding='utf-8'))
def exact(a,b,reason):
    a=np.asarray(a);b=np.asarray(b)
    assert a.shape==b.shape,reason+': shape'
    if a.dtype.kind in 'f' and b.dtype.kind in 'f':ok=np.equal(a,b)|(np.isnan(a)&np.isnan(b))
    else:ok=a==b
    assert bool(np.all(ok)),reason
def date_order(frame):
    assert frame.date.is_unique,'DUPLICATE_DATE'
    assert frame.date.is_monotonic_increasing,'DATE_ORDER'
def interval_invariants(lower,upper,width,score):
    assert np.all(np.asarray(lower)<=np.asarray(upper)),'INTERVAL_ORDER'
    assert np.all(np.asarray(score)>=np.asarray(width)),'IS_BELOW_WIDTH'
def fixed_invariant(q):assert np.all(np.asarray(q)==q.iloc[0]),'FIXED_Q_CHANGED'
def quantile_invariant(scores,q):
    expected=np.sort(np.asarray(scores),kind='stable')[math.ceil((len(scores)+1)*.9)-1]
    assert q==expected,'CONFORMAL_RANK'
def history_invariant(dates,target,expected):
    assert len(dates)==252,'HISTORY_LENGTH'
    assert all(d<target for d in dates),'CURRENT_SCORE_LEAKAGE'
    assert list(dates)==list(expected),'HISTORY_NOT_PREVIOUS_252'
def paired_invariant(sample,reference,index):
    assert len(reference)==1658,'NOT_FULL_TEST_SOURCE'
    exact(sample,reference[index],'PAIR_OR_H_INDEX_MISMATCH')

class OriginalBaselineTests(unittest.TestCase):
    def test_preprocessing_and_splits(self):
        data=read('data/processed/analysis_dataset.csv');date_order(data)
        self.assertEqual(len(data),2412);self.assertFalse(data.isna().any().any())
        for split,n in [('training',502),('calibration',252),('test',1658)]:
            self.assertEqual(int(data.split.eq(split).sum()),n)
        raw=pd.read_csv(SAVED/'data/raw/SP500_fred.csv',dtype=str,keep_default_na=False)
        close=pd.to_numeric(raw.SP500,errors='coerce');valid=close.notna()
        market=pd.DataFrame({'date':raw.loc[valid,'observation_date'],'price':close[valid]}).reset_index(drop=True)
        returns=100*np.log(market.price).diff()
        bydate=market.assign(ret=returns,lag=returns.shift(1),origin=market.date.shift(1)).set_index('date')
        reference=bydate.loc[data.date]
        exact(data.return_pct,reference.ret,'RETURN_LOG_DIFF')
        exact(data.return_lag1_pct,reference.lag,'RETURN_LAG')
        exact(data.forecast_origin_date,reference.origin,'ORIGIN_MARKET_DATE')
        vr=pd.read_csv(SAVED/'data/raw/VIXCLS_fred.csv',dtype=str,keep_default_na=False)
        vix=pd.Series(pd.to_numeric(vr.VIXCLS,errors='coerce').to_numpy(),index=vr.observation_date)
        exact(data.vix_lag1,vix.loc[data.forecast_origin_date],'VIX_MARKET_DATE')
        for threshold in [25,30,35]:exact(data[f'high_vix_{threshold}'],data.vix_lag1.ge(threshold),'VIX_CLASSIFICATION')
        self.assertEqual(int(data.loc[data.split=='test','high_vix_30'].sum()),149)

    def test_ar1_saved_coefficients_all_predictions(self):
        model=js('models/ols/fixed_coefficients.json')
        self.assertEqual(model['training_n'],502)
        for split in ['calibration','test']:
            frame=read(f'reports/ols/{split}_point_predictions.csv')
            prediction=model['coefficients']['const']+model['coefficients']['return_lag1_pct']*frame.return_lag1_pct
            exact(prediction,frame.predicted_return_pct,'AR1_ALL_PREDICTIONS')
        path=VALIDATION/'logs/11_result.json'
        self.assertTrue(path.exists(),'OLS isolated execution not completed')
        record=json.loads(path.read_text(encoding='utf-8'));calls=record['fit_calls']
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['n'],502)
        training=read('data/processed/analysis_dataset.csv').query("split == 'training'")
        exact(np.array(calls[0]['row_labels'],dtype='datetime64[ns]'),pd.to_datetime(training.date).to_numpy(),'OLS_TRAINING_ONLY')

    def test_fixed_quantile_and_all_endpoints(self):
        scores=read('reports/fixed_conformal/calibration_scores_2019.csv')
        threshold=js('models/fixed_conformal/fixed_threshold.json')
        self.assertEqual(len(scores),252);self.assertTrue(scores.date.str.startswith('2019').all())
        k=math.ceil((len(scores)+1)*.9);self.assertEqual(k,228)
        q=np.sort(scores.score_pct.to_numpy(),kind='stable')[k-1]
        self.assertEqual(q,threshold['q_fixed_pct'])
        frame=read('reports/fixed_conformal/test_fixed_intervals.csv')
        fixed_invariant(frame.q_fixed_pct)
        exact(frame.q_fixed_pct,np.full(len(frame),q),'FIXED_QUANTILE')
        exact(frame.predicted_return_pct-q,frame.lower_fixed_pct,'FIXED_LOWER')
        exact(frame.predicted_return_pct+q,frame.upper_fixed_pct,'FIXED_UPPER')

    def test_rolling_all_histories_quantiles_and_no_current_score(self):
        cal=read('reports/fixed_conformal/calibration_scores_2019.csv')
        test=read('reports/rolling_conformal/test_scores.csv')
        history=read('reports/rolling_conformal/history_audit.csv')
        interval=read('reports/rolling_conformal/test_rolling_intervals.csv')
        dates=np.r_[cal.date.to_numpy(),test.date.to_numpy()]
        scores=np.r_[cal.score_pct.to_numpy(),test.test_score_pct.to_numpy()]
        for i,row in history.iterrows():
            past=dates[i:i+252];history_invariant(past,row.date,dates[i:i+252])
            self.assertEqual(past[0],row.oldest_score_date);self.assertEqual(past[-1],row.latest_score_date)
            self.assertEqual(past[-1],row.forecast_origin_date)
            self.assertEqual(row.training_score_count,0)
            self.assertEqual(row.calibration_score_count,max(252-i,0));self.assertEqual(row.past_test_score_count,min(i,252))
            self.assertEqual(int(row.history_start_sequence_index),i);self.assertEqual(int(row.history_end_sequence_index),i+251)
            q=np.sort(scores[i:i+252],kind='stable')[227]
            self.assertEqual(q,row.q_rolling_pct);self.assertEqual(q,interval.q_rolling_pct.iloc[i])
            # Altering the excluded current score cannot affect this history quantile.
            if i in [0,828,1657]:
                altered=scores.copy();altered[i+252]+=100
                self.assertEqual(np.sort(altered[i:i+252],kind='stable')[227],q)
                self.assertNotEqual(altered[i+1:i+253][-1],scores[i+1:i+253][-1])
        fixed=read('reports/fixed_conformal/test_fixed_intervals.csv')
        self.assertEqual(interval.q_rolling_pct.iloc[0],fixed.q_fixed_pct.iloc[0])
        exact(interval.predicted_return_pct-interval.q_rolling_pct,interval.lower_rolling_pct,'ROLLING_LOWER')
        exact(interval.predicted_return_pct+interval.q_rolling_pct,interval.upper_rolling_pct,'ROLLING_UPPER')
        dynamic=VALIDATION/'logs/13_result.json'
        self.assertTrue(dynamic.exists(),'Rolling timing trace missing')
        trace=json.loads(dynamic.read_text(encoding='utf-8'))['rolling_timing']
        self.assertTrue(all(trace[str(i)].get('actual_read_after_interval') for i in range(1658)))

    def test_daily_metrics_and_logical_invariants(self):
        d=read('reports/evaluation/daily_test_evaluation.csv');y=d.return_pct.to_numpy()
        for method in ['fixed','rolling']:
            lo=d[f'lower_{method}_pct'].to_numpy();up=d[f'upper_{method}_pct'].to_numpy()
            width=up-lo;score=width+20*np.maximum(lo-y,0)+20*np.maximum(y-up,0)
            interval_invariants(lo,up,width,score)
            exact(width,d[f'width_{method}_return_pp'],'DAILY_WIDTH')
            exact(score,d[f'interval_score_{method}_return_pp'],'DAILY_IS')
            exact((lo<=y)&(y<=up),d[f'covered_{method}'],'DAILY_COVERAGE')
        exact(d.high_vix_30,d.vix_lag1.ge(30),'DAILY_H')

    def test_saved_bootstrap_all_replicates_and_CI(self):
        folders=['reports/bootstrap','reports/bootstrap_sensitivity/block_5','reports/bootstrap_sensitivity/block_20',
            'reports/vix_threshold_sensitivity/threshold_25','reports/vix_threshold_sensitivity/threshold_35',
            'reports/rolling_window_sensitivity/window_63','reports/rolling_window_sensitivity/window_126']
        metric={'coverage_difference':'coverage_difference_rolling_minus_fixed','mean_width_difference':'mean_width_difference_rolling_minus_fixed_return_pp',
            'mean_interval_score_difference':'mean_interval_score_difference_rolling_minus_fixed_return_pp'}
        for folder in folders:
            rep=read(folder+'/replicate_statistics.csv');self.assertEqual(rep.replicate.nunique(),5000)
            self.assertFalse(rep[['replicate','group']].duplicated().any())
            pivot=rep.pivot(index='replicate',columns='group',values='n')
            exact(pivot.high_vix+pivot.non_high_vix,np.full(5000,1658),'BOOTSTRAP_N_SUM')
            self.assertTrue(rep.n.gt(0).all())
            for m in ['fixed','rolling']:
                self.assertTrue(rep[f'coverage_{m}'].between(0,1).all())
                self.assertTrue((rep[f'mean_interval_score_{m}_return_pp']>=rep[f'mean_width_{m}_return_pp']).all())
            ci=read(folder+'/difference_confidence_intervals.csv')
            for r in ci.itertuples():
                values=rep.loc[rep.group==r.group,metric[r.metric]]
                exact(np.quantile(values,[.025,.975],method='linear'),[r.ci_lower,r.ci_upper],'SAVED_ALL_REPLICATE_CI')

    def test_bootstrap_recorded_full_index_trace(self):
        execution=json.loads((VALIDATION/'baseline_execution.json').read_text(encoding='utf-8'))
        for stage in ['15','16','17','18']:
            record=next((x for x in execution['stages'] if x['stage']==stage),None)
            self.assertIsNotNone(record,'full trace execution missing: '+stage)
            self.assertEqual(record['status'],'PASS')
            draws=json.loads((VALIDATION/'traces'/f'{stage}_draws.json').read_text(encoding='utf-8'))
            trace=np.load(VALIDATION/'traces'/f'{stage}_indices.npz',allow_pickle=False)
            all_indices=trace['indices'];all_starts=trace['starts'];offsets=trace['start_offsets']
            self.assertEqual(len(draws),5003*(2 if stage=='16' else 1))
            seen={}
            for j,draw in enumerate(draws):
                starts=all_starts[offsets[j]:offsets[j+1]]
                length=draw['L'];number=seen.get(length,0)
                # An independent generator verifies the recorded stream without touching the old run's RNG.
                if number in [0,5000]:rng=np.random.default_rng(202701)
                state_hash=lambda:hashlib.sha256(json.dumps(rng.bit_generator.state,sort_keys=True).encode()).hexdigest()
                self.assertEqual(state_hash(),draw['rng_before_sha256'])
                reference_starts=rng.integers(0,1658-length+1,size=math.ceil(1658/length),dtype=np.int64)
                exact(starts,reference_starts,'RNG_STARTS')
                self.assertEqual(state_hash(),draw['rng_after_sha256'])
                seen[length]=number+1
                idx=(starts[:,None]+np.arange(draw['L'])).ravel()[:1658]
                exact(idx,all_indices[j],'ALL_INDEX_RECONSTRUCTION')
                self.assertEqual(hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest(),draw['index_sha256'])
            self.assertGreater(record['paired_sample_checks'],5000)

    def test_sensitivity_settings_and_window_histories(self):
        d=read('reports/evaluation/daily_test_evaluation.csv')
        counts=read('reports/vix_threshold_sensitivity/threshold_summary.csv')
        for threshold,n in [(25,348),(30,149),(35,59)]:
            self.assertEqual(int(d.vix_lag1.ge(threshold).sum()),n)
            self.assertEqual(int(counts.loc[counts.vix_threshold==threshold,'high_vix_n'].iloc[0]),n)
        cal=read('reports/fixed_conformal/calibration_scores_2019.csv');test=read('reports/rolling_conformal/test_scores.csv')
        allscores=np.r_[cal.score_pct,test.test_score_pct]
        for w in [63,126]:
            interval=read(f'reports/rolling_window_sensitivity/window_{w}/test_rolling_intervals.csv')
            rank=math.ceil((w+1)*.9)
            q=np.array([np.sort(allscores[252+i-w:252+i],kind='stable')[rank-1] for i in range(1658)])
            exact(q,interval.q_rolling_pct,'SENSITIVITY_WINDOW_QUANTILES')
            self.assertEqual(len(interval),1658)

    def test_additional_bootstrap_and_HAC_separate(self):
        folder='audits/2026-10-24_additional_statistical_audit'
        metric=['coverage_difference','mean_width_difference','mean_interval_score_difference']
        for method,lengths in [('mbb',[40,60,120]),('sb',[10,40,60,120])]:
            for length in lengths:
                rep=read(f'{folder}/tables/{method}_{length}_replicates.csv')
                self.assertEqual(len(rep),20000);self.assertEqual(rep.replicate.nunique(),5000)
                p=rep.pivot(index='replicate',columns='group',values='n')
                exact(p.high_vix+p.non_high_vix,np.full(5000,1658),'ADDITIONAL_GROUP_COUNTS')
                summary=read(f'{folder}/tables/new_bootstrap_summary.csv')
                selected=summary[(summary.method==method)&(summary.length==length)]
                for row in selected.itertuples():
                    values=rep.loc[rep.group==row.group,row.metric]
                    exact(np.quantile(values,[.025,.975],method='linear'),[row.ci_lower,row.ci_upper],'ADDITIONAL_SAVED_CI')
                    self.assertEqual(values.to_numpy().std(ddof=1),row.standard_error)
        hac=read(folder+'/tables/hac_summary.csv')
        self.assertEqual(len(hac),72)
        self.assertTrue(np.isfinite(hac[['point_estimate','standard_error','ci_lower','ci_upper']]).all().all())
        self.assertFalse(hac.negative_variance_flag.any())
        self.assertEqual(sorted(hac.length.unique().tolist()),[5,10,20,40,60,120])
        logpath=VALIDATION/'logs/additional_result.json'
        self.assertTrue(logpath.exists(),'additional isolated numerical phases not completed')
        self.assertEqual(json.loads(logpath.read_text(encoding='utf-8'))['status'],'PASS')

    def test_additional_full_rng_trace(self):
        logpath=VALIDATION/'traces/additional_draws.json';self.assertTrue(logpath.exists(),'additional trace missing')
        draws=json.loads(logpath.read_text(encoding='utf-8'))
        arrays=np.load(VALIDATION/'traces/additional_indices.npz',allow_pickle=False)
        indices=arrays['indices'];starts_all=arrays['starts'];offsets=arrays['start_offsets'];seen={}
        self.assertEqual(len(draws),35021)
        for j,row in enumerate(draws):
            key=(row['function'],row['L']);number=seen.get(key,0)
            if number in [0,5000]:rng=np.random.default_rng(20261024)
            self.assertEqual(hashlib.sha256(json.dumps(rng.bit_generator.state,sort_keys=True).encode()).hexdigest(),row['rng_before_sha256'])
            if row['function']=='draw_mbb':
                starts=rng.integers(0,1658-row['L']+1,size=math.ceil(1658/row['L']),dtype=np.int64)
                expected=(starts[:,None]+np.arange(row['L'],dtype=np.int64)).ravel()[:1658]
            else:
                chunks=[];starts=[];n=0
                while n<1658:
                    s=int(rng.integers(0,1658));size=int(rng.geometric(1/row['L']));retained=min(size,1658-n)
                    starts.append(s);chunks.append((s+np.arange(retained))%1658);n+=retained
                expected=np.concatenate(chunks)
            exact(starts,starts_all[offsets[j]:offsets[j+1]],'ADDITIONAL_RNG_STARTS')
            exact(expected,indices[j],'ADDITIONAL_ALL_INDICES')
            self.assertEqual(hashlib.sha256(json.dumps(rng.bit_generator.state,sort_keys=True).encode()).hexdigest(),row['rng_after_sha256'])
            seen[key]=number+1

    def test_baseline_all_stages_numeric_identity(self):
        record=json.loads((VALIDATION/'baseline_execution.json').read_text(encoding='utf-8'))
        self.assertTrue(record['completed'],'old baseline stages incomplete')
        self.assertIsNone(record['stopped_at'])
        self.assertTrue(all(r['status']=='PASS' for r in record['stages']))
        comparisons=json.loads((VALIDATION/'comparisons/all_outputs.json').read_text(encoding='utf-8'))
        self.assertTrue(all(r['passed'] for r in comparisons))
        pixels=json.loads((VALIDATION/'comparisons/figure_pixels.json').read_text(encoding='utf-8'))
        self.assertTrue(all(r['pixels_equal'] for r in pixels),'FIGURE_PIXELS_CHANGED')

    def test_mutation_detection_expected_reasons(self):
        d=read('reports/evaluation/daily_test_evaluation.csv')
        cases=[]
        def expect(name,reason,call):
            with self.assertRaisesRegex(AssertionError,reason):call()
            cases.append({'mutation':name,'expected_reason':reason,'detected':True})
        bad=d.copy();bad.loc[1,'date']=bad.loc[0,'date']
        expect('duplicate date','DUPLICATE_DATE',lambda:date_order(bad))
        bad_order=d.iloc[::-1].copy();expect('reverse order','DATE_ORDER',lambda:date_order(bad_order))
        expect('reversed endpoints','INTERVAL_ORDER',lambda:interval_invariants([2.],[1.],[1.],[1.]))
        expect('IS below width','IS_BELOW_WIDTH',lambda:interval_invariants([0.],[1.],[1.],[.5]))
        q=d.q_fixed_pct.copy();q.iloc[10]+=1
        expect('vary fixed q','FIXED_Q_CHANGED',lambda:fixed_invariant(q))
        cal=read('reports/fixed_conformal/calibration_scores_2019.csv');dates=cal.date.tolist();contaminated=dates[1:]+[d.date.iloc[0]]
        expect('wrong conformal quantile','CONFORMAL_RANK',lambda:quantile_invariant(cal.score_pct,d.q_fixed_pct.iloc[0]+1))
        expect('current score enters history','CURRENT_SCORE_LEAKAGE',lambda:history_invariant(contaminated,d.date.iloc[0],dates))
        expect('wrong previous window','HISTORY_NOT_PREVIOUS_252',lambda:history_invariant(dates[::-1],d.date.iloc[0],dates))
        wrong_h=d.high_vix_30.copy();wrong_h.iloc[0]=not wrong_h.iloc[0]
        expect('wrong High classification','VIX_CLASSIFICATION',lambda:exact(wrong_h,d.vix_lag1.ge(30),'VIX_CLASSIFICATION'))
        cols=['high_vix_30','covered_fixed','covered_rolling','width_fixed_return_pp','width_rolling_return_pp','interval_score_fixed_return_pp','interval_score_rolling_return_pp']
        reference=d[cols].to_numpy(float);idx=np.arange(1658)[::-1];sample=reference[idx].copy()
        sample[:,2]=np.roll(sample[:,2],1)
        expect('break rolling pair','PAIR_OR_H_INDEX_MISMATCH',lambda:paired_invariant(sample,reference,idx))
        sample=reference[idx].copy();sample[:,0]=reference[:,0]
        expect('H uses different index','PAIR_OR_H_INDEX_MISMATCH',lambda:paired_invariant(sample,reference,idx))
        expect('compress High before bootstrap','NOT_FULL_TEST_SOURCE',lambda:paired_invariant(reference[:149],reference[d.high_vix_30],np.arange(149)))
        good_rng=np.random.default_rng(202701);bad_rng=np.random.default_rng(202702)
        expect('change RNG seed','RNG_STARTS',lambda:exact(good_rng.integers(0,1649,size=166,dtype=np.int64),bad_rng.integers(0,1649,size=166,dtype=np.int64),'RNG_STARTS'))
        (VALIDATION/'mutation_results.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':unittest.main(verbosity=2)
