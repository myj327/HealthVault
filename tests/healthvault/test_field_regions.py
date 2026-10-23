import io
import tempfile
import unittest
from pathlib import Path

from healthvault.web.evidence import layout_blocks
from healthvault.web.extraction import build_natural_preview
from healthvault.web.documents import extract_pdf_document
from healthvault.application.confirm_extracted_report import confirm_extracted_report
from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository


def block(text, x, y, width=70):
    return {'text':text, 'page':1, 'polygon':[[x,y],[x+width,y],[x+width,y+12],[x,y+12]], 'confidence':.99,
            'page_width':600, 'page_height':800, 'coordinate_space':'display_pixels'}


class FieldRegionTests(unittest.TestCase):
    def test_split_table_cells_and_repeated_values_bind_to_correct_rows(self):
        cells = [block('报告日期: 2026-10-01', 10,10,200), block('空腹血糖',10,50), block('5.2',100,50),
                 block('mmol/L',180,50), block('3.9–6.1',290,50), block('血红蛋白',10,90), block('132',100,90),
                 block('g/L',180,90), block('115–150',290,90)]
        text, blocks = layout_blocks(cells)
        preview = build_natural_preview('source', text, {}, blocks=blocks)
        self.assertTrue(preview['can_confirm'], preview)
        self.assertEqual(2, len(preview['indicators']))
        for item in preview['indicators']:
            field = item['evidence']['fields']['value']
            self.assertEqual(2, field['regions'][0]['column_index'])
            self.assertEqual(field['text'], text[field['span_start']:field['span_end']])
            self.assertEqual(item['evidence']['regions'][0]['row_index'], field['regions'][0]['row_index'])
        with tempfile.TemporaryDirectory() as directory:
            repository = SQLiteHealthRecordRepository(Path(directory)/'records.sqlite3')
            result = confirm_extracted_report('user', 'source', text, preview, repository)
            self.assertTrue(result['confirmed'])
            reports = repository.list_timeline('user')
            self.assertTrue(reports)
            self.assertEqual(2, reports[0].to_dict()['indicators'][0]['evidence']['fields']['value']['regions'][0]['column_index'])

    def test_manual_text_correction_preserves_original_cell_region(self):
        original, blocks = layout_blocks([block('报告日期: 2026-10-01',10,10,200), block('空腹血糖 6.2 mmol/L 3.9–6.1',10,50,400)])
        corrected = original.replace('6.2', '5.2')
        preview = build_natural_preview('source', corrected, {}, blocks=blocks, original_text=original)
        field = preview['indicators'][0]['evidence']['fields']['value']
        self.assertEqual('5.2', field['text'])
        self.assertTrue(field['regions'][0]['corrected_text'])

    def test_canonical_web_preview_keeps_regions_after_confirmation(self):
        import hashlib
        from fastapi.testclient import TestClient
        from healthvault.web.api import create_app
        from healthvault.ui.controller import HealthVaultUIController
        text = 'report_id: region-legacy\nreport_date: 2026-10-01\nfacility: Synthetic hospital\nresult|fasting_glucose|5.2|mmol/L|3.9|6.1'
        cells = [block(line, 10, index*30, 450) for index,line in enumerate(text.splitlines(),1)]
        class Controller(HealthVaultUIController):
            def process_image(self, raw, filename):
                return {'text':text, 'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                        'blocks':cells, 'confidence':{'average':.99,'minimum':.99,'low_confidence_blocks':[]}}
        with tempfile.TemporaryDirectory() as directory:
            controller = Controller(Path(directory)/'records.sqlite3')
            with TestClient(create_app(controller, skill_evolution_root=Path(directory)/'evolution',
                                       knowledge_release_root=Path(directory)/'knowledge')) as client:
                store = client.app.state.auth_store
                user = store.register('region-user','synthetic-password-123',user_id='region-user')
                token, csrf = store.create_session(user)
                client.cookies.set('hv_session',token); client.headers['X-CSRF-Token'] = csrf
                draft = client.post('/api/drafts/image',data={'profile_id':user.profile_id},
                                    files={'file':('synthetic.png',b'\x89PNG\r\n\x1a\nsynthetic','image/png')}).json()
                response = client.put(f"/api/drafts/{draft['token']}", json={
                    'profile_id':user.profile_id, 'report_text':text, 'reviewed':True})
                self.assertEqual(200,response.status_code,response.text)
                draft = response.json()
                field = draft['preview']['indicators'][0]['evidence']['fields']['value']
                self.assertEqual('5.2',field['text'])
                self.assertEqual(1,len(field['regions']))
                response = client.post(f"/api/drafts/{draft['token']}/confirm",json={
                    'profile_id':user.profile_id,'report_text':text,
                    'preview_sha256':draft['preview_sha256'],'acknowledged':True})
                self.assertEqual(200,response.status_code,response.text)
                self.assertTrue(response.json()['confirmed'])
                restored = controller.record_repository.list_timeline(user.profile_id)[0].to_dict()
                self.assertEqual(field,restored['indicators'][0]['evidence']['fields']['value'])

    def test_two_findings_on_one_line_keep_independent_sides_and_sizes(self):
        text = '报告日期: 2026-10-01\n甲状腺左叶结节 6×4 mm，右叶结节 8×5 mm'
        preview = build_natural_preview('source', text, {})
        self.assertEqual(2, len(preview['imaging_findings']), preview)
        self.assertEqual(['left','right'], [item['laterality'] for item in preview['imaging_findings']])
        self.assertEqual([[6.0,4.0],[8.0,5.0]], [item['dimensions_mm'] for item in preview['imaging_findings']])
        self.assertNotEqual(preview['imaging_findings'][0]['evidence']['span_start'], preview['imaging_findings'][1]['evidence']['span_start'])

    def test_deskew_accepts_flat_and_nested_opencv_line_arrays(self):
        import numpy as np
        from PIL import Image
        from unittest.mock import patch
        from healthvault.ocr.preprocess import prepare_image
        image = Image.new('RGB',(400,200),'white'); stream = io.BytesIO(); image.save(stream,format='PNG')
        lines = np.asarray([[0,0,100,2],[10,10,110,12],[20,20,120,22]],dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory:
            results = []
            for index, shape in enumerate((lines,lines.reshape(-1,1,4))):
                with patch('cv2.HoughLinesP',return_value=shape):
                    results.append(prepare_image(stream.getvalue(),Path(directory)/f'{index}.png'))
            self.assertEqual('prepared',results[0]['status'])
            self.assertGreater(results[0]['deskew_degrees'],0)
            self.assertEqual(results[0],results[1])

    def test_multiple_scanned_regions_preserve_native_title_and_coordinates(self):
        import pymupdf
        from PIL import Image
        image = Image.new('RGB', (200,100), 'white'); stream = io.BytesIO(); image.save(stream,format='PNG')
        document = pymupdf.open(); page = document.new_page(width=600,height=800)
        page.insert_text((20,25),'Hospital report native title with enough selectable text')
        page.insert_image(pymupdf.Rect(20,100,280,300), stream=stream.getvalue())
        page.insert_image(pymupdf.Rect(310,100,570,300), stream=stream.getvalue())
        raw = document.tobytes(); document.close()
        calls = []
        def ocr(data, filename):
            calls.append(filename)
            return {'text':f'scanned-{len(calls)}', 'blocks':[block(f'scanned-{len(calls)}',10,10)], 'confidence':{}}
        result = extract_pdf_document(raw, ocr)
        self.assertEqual(2,len(calls))
        self.assertIn('Hospital report', result['text'])
        self.assertIn('scanned-1', result['text']); self.assertIn('scanned-2', result['text'])
        regions = [entry for entry in result['blocks'] if entry['text'].startswith('scanned')]
        self.assertGreater(regions[1]['polygon'][0][0],300)
        for entry in result['blocks']:
            self.assertEqual(entry['text'],result['text'][entry['span_start']:entry['span_end']])

if __name__ == '__main__':
    unittest.main()
