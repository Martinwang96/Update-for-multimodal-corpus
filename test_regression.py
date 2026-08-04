#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""关键回归测试。"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import app as flask_app
from analyzers import BodyAnalyzer, FaceAnalyzer


if 'api_boom_for_test' not in flask_app.app.view_functions:
    @flask_app.app.route('/api/_boom-for-test')
    def api_boom_for_test():
        raise RuntimeError('boom')


_BODY_PIPELINE_PATH = Path(__file__).resolve().parent / '躯体' / '综合处理-2d.py'
_BODY_PIPELINE_SPEC = importlib.util.spec_from_file_location('body_pipeline_test', str(_BODY_PIPELINE_PATH))
body_pipeline = importlib.util.module_from_spec(_BODY_PIPELINE_SPEC)
assert _BODY_PIPELINE_SPEC.loader is not None
_BODY_PIPELINE_SPEC.loader.exec_module(body_pipeline)

_DISPLACEMENT_PATH = Path(__file__).resolve().parent / '躯体' / '位移-2d.py'
_DISPLACEMENT_SPEC = importlib.util.spec_from_file_location('body_displacement_test', str(_DISPLACEMENT_PATH))
displacement_module = importlib.util.module_from_spec(_DISPLACEMENT_SPEC)
assert _DISPLACEMENT_SPEC.loader is not None
_DISPLACEMENT_SPEC.loader.exec_module(displacement_module)


class RegressionTests(unittest.TestCase):
    def setUp(self):
        flask_app.app.config.update(TESTING=False, PROPAGATE_EXCEPTIONS=False)
        self.client = flask_app.app.test_client()
        with flask_app.TASK_LOCK:
            flask_app.TASKS.clear()
        with flask_app.PREVIEW_LOCK:
            flask_app.PREVIEWS.clear()
        with flask_app.UPLOAD_LOCK:
            flask_app.UPLOADS.clear()

    def test_run_task_marks_payload_error_as_task_error(self):
        task_id = 't_error_payload'
        with flask_app.TASK_LOCK:
            flask_app.TASKS[task_id] = {'status': 'pending', 'result': None, 'error': None}

        flask_app.run_task(task_id, lambda: {'status': 'error', 'message': 'bad result'})

        with flask_app.TASK_LOCK:
            task = dict(flask_app.TASKS[task_id])

        self.assertEqual(task['status'], 'error')
        self.assertIn('bad result', task.get('error') or '')
        self.assertIsNone(task.get('result'))

    def test_face_modules_do_not_advertise_gaze(self):
        modules = FaceAnalyzer().list_modules()
        self.assertNotIn('gaze', modules)

    def test_final_analysis_rejects_disabled_gaze_subtype(self):
        with flask_app.PREVIEW_LOCK:
            flask_app.PREVIEWS['pv_gaze'] = {
                'csv_path': '/tmp/fake.csv',
                'video_path': None,
                'kind': 'csv',
                'module': 'face',
                'subtype': 'gaze',
                'axis': '',
                'result': {'status': 'ok'},
            }

        resp = self.client.post('/api/analyze/final', data={'preview_id': 'pv_gaze'})

        self.assertEqual(resp.status_code, 400)
        self.assertTrue(resp.is_json)
        self.assertIn('暂未开放', resp.get_json()['error'])

    def test_api_exceptions_return_json_for_api_routes(self):
        resp = self.client.get('/api/_boom-for-test')

        self.assertEqual(resp.status_code, 500)
        self.assertTrue(resp.is_json)
        payload = resp.get_json()
        self.assertIn('error', payload)
        self.assertNotIn('<!doctype', json.dumps(payload).lower())

    def test_body_analyzer_forces_serial_mode(self):
        analyzer = BodyAnalyzer()
        captured = {}

        class FakePipeline:
            @staticmethod
            def run_pipeline(config, progress_callback=None):
                captured.update(config)
                return {'results': []}

        with mock.patch('analyzers.body_analyzer._load_pipeline_module', return_value=FakePipeline()):
            result = analyzer.analyze_from_json('/tmp/fake.json', '/tmp/out', params={'parallel': True})

        self.assertEqual(result['status'], 'ok')
        self.assertIn('parallel', captured)
        self.assertFalse(captured['parallel'])

    def test_body_preview_returns_default_params_for_json_input(self):
        with flask_app.UPLOAD_LOCK:
            flask_app.UPLOADS['f_body_json'] = {
                'path': '/tmp/fake.json',
                'filename': 'fake.json',
                'kind': 'json',
                'openface_csv': None,
            }

        expected = {'tilt': {'smooth_window': 9}, 'shrug': {}, 'displacement': {}, 'rotation': {}}
        fake_analyzer = mock.Mock()
        fake_analyzer.default_params.return_value = expected

        with mock.patch('app.get_body_analyzer', return_value=fake_analyzer):
            resp = self.client.post('/api/preview', data={'file_id': 'f_body_json', 'module': 'body'})

        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json()
        self.assertEqual(payload['preview']['recommended'], expected)
        self.assertIn('可直接调整', payload['preview']['note'])

    def test_final_analysis_passes_body_params_json(self):
        with flask_app.PREVIEW_LOCK:
            flask_app.PREVIEWS['pv_body_json'] = {
                'csv_path': '/tmp/fake.json',
                'video_path': None,
                'kind': 'json',
                'module': 'body',
                'subtype': '',
                'axis': '',
                'result': {'status': 'ok'},
            }

        captured = {}
        fake_analyzer = mock.Mock()
        fake_analyzer.analyze_from_json = object()

        def fake_start_task(func, *args, **kwargs):
            captured['func'] = func
            captured['args'] = args
            captured['kwargs'] = kwargs
            return 'task_body_1'

        with mock.patch('app.get_body_analyzer', return_value=fake_analyzer), \
             mock.patch('app.start_task', side_effect=fake_start_task):
            resp = self.client.post('/api/analyze/final', data={
                'preview_id': 'pv_body_json',
                'body_params_json': json.dumps({'tilt': {'smooth_window': 11}, 'displacement': {'threshold_move_override': 12.5}}),
            })

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()['task_id'], 'task_body_1')
        self.assertIs(captured['func'], fake_analyzer.analyze_from_json)
        self.assertEqual(captured['args'][0], '/tmp/fake.json')
        self.assertEqual(captured['args'][2], {'tilt': {'smooth_window': 11}, 'displacement': {'threshold_move_override': 12.5}})

    def test_displacement_fill_helpers_work_with_current_pandas(self):
        positions = np.array([[0.0, 0.0], [np.nan, np.nan], [2.0, 2.0]], dtype=float)
        interpolated = displacement_module.interpolate_invalid_frames_2d(
            positions,
            np.array([True, False, True], dtype=bool),
            limit_gap_frames=3,
        )
        smoothed = displacement_module.smooth_positions_2d(interpolated, window_length=3, poly_order=1)
        baseline_x, deviation_x = displacement_module.calculate_baseline_and_deviation_x(smoothed[:, 0], fps=30.0, baseline_window_sec=1.0)

        self.assertEqual(interpolated.shape, positions.shape)
        self.assertEqual(smoothed.shape, positions.shape)
        self.assertEqual(baseline_x.shape[0], positions.shape[0])
        self.assertEqual(deviation_x.shape[0], positions.shape[0])

    def test_tilt_all_nan_returns_soft_empty_result(self):
        task = {
            'json': '/tmp/fake.json',
            'out': tempfile.mkdtemp(prefix='tilt-soft-'),
            'fps': 30.0,
            'conf': 0.3,
            'plots': False,
            'params': body_pipeline.build_default_params()['tilt'],
        }

        class FakeTiltModule:
            DEFAULT_REQUIRED_KP_INDICES = (5, 6, 11, 12)

            @staticmethod
            def load_required_keypoints_2d(jp, kp_indices, conf):
                coords = np.zeros((60, 4, 2), dtype=float)
                valid = np.ones(60, dtype=bool)
                return coords, None, 60, valid

            @staticmethod
            def interpolate_keypoint_positions(coords):
                arr = np.zeros((60, 2), dtype=float)
                return arr.copy(), arr.copy(), arr.copy(), arr.copy()

            @staticmethod
            def calculate_torso_tilt_angle_2d(mh, ms):
                return np.full(60, np.nan)

        with mock.patch.object(body_pipeline, 'load_mod', return_value=FakeTiltModule()):
            result = body_pipeline.run_tilt(task)

        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['event_count'], 0)
        self.assertEqual(result['data_status'], 'insufficient')
        self.assertIn('无法形成有效躯干角度', result['note'])


if __name__ == '__main__':
    unittest.main()
