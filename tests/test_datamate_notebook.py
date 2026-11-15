"""Exercise the actual notebook functions on synthetic data, without GPU/downloads."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts import datamate_baseline as dm


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/datamate_qwen_kaggle.ipynb'
RULE = 'Giữ mọi dòng; tổng tiền không loại trùng.'
DOMAIN = json.dumps({'meaning': 'Tổng doanh thu', 'rules': [
    {'source': 'rulebook', 'quote': RULE, 'application': 'Cộng mọi dòng'}],
    'unknowns': []}, ensure_ascii=False)
PROBE = '''def answer(tables, as_of_date, calendar):
    return {'type': 'string', 'answer': str(len(tables['sales'])) + ' rows'}
'''
PLAN = json.dumps({'tables': {'sales': ['amount']}, 'answer_type': 'number',
                   'steps': ['1. Parse amount', '2. Sum all rows'], 'checks': ['Row count is 3']})
CODE = '''def answer(tables, as_of_date, calendar):
    df = tables['sales']
    if len(df) != 3:
        raise ValueError('Expected three rows')
    print('rows:', len(df))
    return {'type': 'number', 'answer': df['amount'].map(parse_money).sum()}
'''


def review(decision='ready'):
    return json.dumps({'decision': decision, 'evidence': ['3 rows from probe'],
                       'applied_rules': ['Keep all rows'],
                       'missing': [] if decision == 'ready' else ['Check duplicate policy'],
                       'next_action': 'Read duplicate policy'})


class Model:
    def __init__(self, responses=(), tokens=100):
        self.responses = iter(responses)
        self.calls = []
        self.tokens = tokens

    def count_tokens(self, messages):
        return self.tokens

    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return dm.Generation(response, 100, 30, 0.01)


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        split = root / 'train'
        business = split / 'data/biz_1'
        business.mkdir(parents=True)
        (business / 'README.md').write_text('Sales amounts in VND.', encoding='utf-8')
        (business / 'sales.csv').write_text('amount\n10\n20\n20\n', encoding='utf-8')
        (split / 'rulebook.md').write_text(RULE, encoding='utf-8')
        (split / 'calendar.csv').write_text('date\n2026-01-02\n', encoding='utf-8')
        self.q = {'id': 'q1', 'business_id': 'biz_1', 'question': 'Tổng tiền?', 'as_of_date': '2026-01-02'}
        (split / 'answers.jsonl').write_text(json.dumps({
            'id': 'q1', 'type': 'number', 'answer': 50, 'secret': 'GOLD_NEVER_IN_PROMPT'}) + '\n')
        self.ns = {'dm': dm}
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if set(cell['metadata'].get('tags', [])) & {'config', 'definitions'}:
                exec(compile(''.join(cell['source']), str(NOTEBOOK), 'exec'), self.ns)
        self.ns.update(dm=dm, pd=dm.pd, display=lambda *args: None,
                       store=dm.DataStore(split), OUTPUT_DIR=root / 'outputs',
                       split_dir=split, config=dm.Config(), MAX_REPAIRS=1,
                       MAX_INSIGHT_ROUNDS=2, EXEC_TIMEOUT=20)
        self.capture = contextlib.redirect_stdout(io.StringIO())
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)

    def run_model(self, responses):
        self.ns['MODEL'] = Model(responses)
        return self.ns['run_question'](self.q)

    def test_actual_cells_success_and_gold_only_after_predictions(self):
        trace = self.run_model([DOMAIN, PROBE, review(), PLAN, CODE])
        self.assertEqual(trace['prediction'], {'id': 'q1', 'type': 'number', 'answer': 50})
        self.assertEqual([e['stage'] for e in trace['events']],
                         ['3_domain', '4_probe_0', '4_review_0', '5_plan', '6_execute'])
        self.assertIn('rows: 3', trace['events'][-1]['execution']['stdout'])
        self.assertNotIn('GOLD_NEVER_IN_PROMPT', json.dumps(self.ns['MODEL'].calls))
        self.assertIn('HỢP ĐỒNG', self.ns['MODEL'].calls[0][-1]['content'])
        self.assertNotIn('Chỉ trả một khối Python', self.ns['MODEL'].calls[0][0]['content'])
        from unittest.mock import patch
        original = dm.read_jsonl

        def checked_read(path):
            self.assertTrue((self.ns['OUTPUT_DIR'] / 'predictions.jsonl').exists())
            return original(path)

        with patch.object(dm, 'read_jsonl', side_effect=checked_read):
            report = self.ns['evaluate_records']([trace])
        self.assertTrue(report.iloc[0]['match'])

    def test_wrong_answer_json_is_repaired_as_plan_not_coerced(self):
        trace = self.run_model([DOMAIN, PROBE, review(), '{"type":"number","answer":0}', PLAN, CODE])
        events = [e for e in trace['events'] if e['stage'] == '5_plan']
        self.assertEqual([e['status'] for e in events], ['error', 'ok'])
        self.assertIn('Selection requires answer_type', events[0]['error'])
        self.assertIn('PHẢN HỒI SAI', events[1]['messages'][-1]['content'])
        self.assertEqual(trace['status'], 'success')

    def test_reread_loop_and_python_starts_in_insights(self):
        trace = self.run_model([DOMAIN, PROBE, review('reread'), DOMAIN,
                                PROBE, review(), PLAN, CODE])
        self.assertEqual(trace['status'], 'success')
        self.assertEqual(sum(e['stage'] == '3_domain' for e in trace['events']), 2)
        self.assertNotIn('execution', trace['events'][0])
        self.assertTrue(next(e for e in trace['events'] if e['stage'] == '4_probe_0')['execution']['ok'])

    def test_domain_rejects_code_and_repairs_without_executing_it(self):
        wrong = json.loads(DOMAIN)
        wrong['probe_code'] = PROBE
        trace = self.run_model([json.dumps(wrong), DOMAIN, PROBE, review(), PLAN, CODE])
        attempts = [e for e in trace['events'] if e['stage'] == '3_domain']
        self.assertEqual([e['status'] for e in attempts], ['error', 'ok'])
        self.assertNotIn('execution', attempts[0])
        self.assertIn('chuyển Python', attempts[0]['error'])
        self.assertEqual(trace['status'], 'success')

    def test_exploration_exhaustion_abstains(self):
        trace = self.run_model([DOMAIN, PROBE, review('explore'), PROBE, review('explore')])
        self.assertEqual(trace['prediction']['type'], 'abstain')
        self.assertEqual(trace['status'], 'failed')
        self.assertNotIn('5_plan', [e['stage'] for e in trace['events']])

    def test_python_failure_repaired_with_traceback(self):
        broken = CODE.replace("tables['sales']", "tables['missing']")
        trace = self.run_model([DOMAIN, PROBE, review(), PLAN, broken, CODE])
        events = [e for e in trace['events'] if e['stage'] == '6_execute']
        self.assertIn('KeyError', events[0]['execution']['error'])
        self.assertEqual(events[1]['status'], 'ok')

    def test_unanswerable_requires_matching_plan_and_remains_success(self):
        plan = json.loads(PLAN)
        plan['answer_type'] = 'unanswerable'
        code = "def answer(tables, as_of_date, calendar):\n    return {'type': 'unanswerable', 'reason': 'Missing required field'}\n"
        trace = self.run_model([DOMAIN, PROBE, review('unanswerable'), json.dumps(plan), code])
        self.assertEqual(trace['status'], 'success')
        self.assertEqual(trace['prediction']['type'], 'unanswerable')

    def test_rerunning_domain_invalidates_old_plan_and_prediction(self):
        self.ns['MODEL'] = Model([DOMAIN])
        state = self.ns['load_question'](self.q)
        state.update(plan=json.loads(PLAN), insights={'decision': 'ready'})
        state['trace'].update(prediction={'type': 'number', 'answer': 0}, status='success')
        self.ns['assimilate_domain'](state)
        self.assertNotIn('plan', state)
        self.assertNotIn('insights', state)
        self.assertNotIn('prediction', state['trace'])
        with self.assertRaises(ValueError):
            self.ns['execute_plan'](state)

    def test_context_and_gpu_errors_stop_without_abstain(self):
        for model, error in [(Model(tokens=999999), dm.ContextOverflow),
                             (Model([dm.RuntimeFailure('CUDA OOM')]), dm.RuntimeFailure)]:
            self.ns['MODEL'] = model
            with self.assertRaises(error):
                self.ns['run_question'](self.q)
            traces = list(self.ns['OUTPUT_DIR'].glob('q1/*/trace.json'))
            trace = json.loads(max(traces, key=lambda p: p.parent.name).read_text(encoding='utf-8'))
            self.assertEqual(trace['status'], 'stopped')
            self.assertNotIn('prediction', trace)

    def test_domain_uses_compact_catalog_and_single_contract(self):
        self.ns['MODEL'] = Model([DOMAIN])
        state = self.ns['load_question'](self.q)
        self.ns['assimilate_domain'](state)
        event = state['trace']['events'][0]
        self.assertTrue(event['compact_catalog'])
        context = self.ns['context_for'](state, compact=True)
        self.assertEqual(context['readme'], state['readme'])
        self.assertEqual(context['rulebook'], RULE)
        self.assertEqual(context['question'], self.q)
        self.assertEqual(set(context['catalog'][0]['columns']), set(state['tables']['sales'].columns))
        self.assertEqual(context['catalog'][0]['row_count'], 3)
        self.assertIn('sales.csv', context['catalog'][0]['source_sheets'])
        messages = event['messages']
        self.assertIn('BƯỚC 3: chỉ trả JSON', messages[0]['content'])
        self.assertNotIn('BƯỚC 3: chỉ trả JSON', messages[1]['content'])

    def test_overflow_rebuilds_catalog_before_generation_and_preserves_repair(self):
        class BudgetModel(Model):
            def count_tokens(self, messages):
                return 12529 if '"sample_rows"' in messages[1]['content'] else 11000

        model = BudgetModel(['invalid JSON', PLAN])
        self.ns['MODEL'] = model
        state = self.ns['load_question'](self.q)
        state['insights'] = {'decision': 'ready', 'evidence': ['EVIDENCE_KEEP']}
        self.ns['formulate_plan'](state)
        for event in state['trace']['events']:
            self.assertEqual(event['context_attempts'], [
                {'compact': False, 'input_tokens': 12529},
                {'compact': True, 'input_tokens': 11000}])
            self.assertIn(RULE, event['messages'][1]['content'])
            self.assertIn('EVIDENCE_KEEP', event['messages'][1]['content'])
        self.assertEqual(len(model.calls), 2)
        self.assertIn('invalid JSON', model.calls[-1][1]['content'])
        self.assertIn('LỖI CẦN SỬA', model.calls[-1][1]['content'])


if __name__ == '__main__':
    unittest.main()
