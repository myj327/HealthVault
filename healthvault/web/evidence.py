"""Server-derived field spans and OCR/PDF region relationships."""
from __future__ import annotations

import difflib
import re
from typing import Any


def layout_blocks(blocks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Group physical rows/cells and preserve text offsets for every block."""
    from healthvault.web.extraction import _indicator_label
    positioned = []
    for index, block in enumerate(blocks):
        polygon = block.get('polygon') or []
        x = min((point[0] for point in polygon), default=0)
        y = min((point[1] for point in polygon), default=index * 20)
        height = max((point[1] for point in polygon), default=y+10) - y
        positioned.append((int(block.get('page', 1)), y, x, max(height, 1), dict(block)))
    rows: list[list[tuple]] = []
    for item in sorted(positioned, key=lambda value: value[:3]):
        if rows and item[0] == rows[-1][0][0] and abs(item[1] - rows[-1][0][1]) <= min(item[3], rows[-1][0][3]) * .45:
            rows[-1].append(item)
        else:
            rows.append([item])
    output = []; text_parts = []; offset = 0
    for row_index, row in enumerate(rows, 1):
        row.sort(key=lambda item: item[2])
        groups: list[list[tuple]] = [[]]
        for item in row:
            text = str(item[4].get('text', '')).strip()
            # Different indicators in side-by-side panels form separate rows.
            if groups[-1] and (_indicator_label(text) or re.match(r'\w+:', text)):
                groups.append([])
            groups[-1].append(item)
        for group in groups:
            for column_index, item in enumerate(group, 1):
                block = item[4]; text = str(block.get('text', '')).strip()
                if not text:
                    continue
                separator = '\n' if text_parts and column_index == 1 else ' ' if text_parts else ''
                text_parts.append(separator + text); offset += len(separator)
                block.update(block_id=f"P{item[0]}-R{row_index}-C{column_index}", row_index=row_index,
                             column_index=column_index, span_start=offset, span_end=offset+len(text), text=text)
                output.append(block); offset += len(text)
    return ''.join(text_parts), output


def bind_preview_evidence(preview: dict[str, Any], text: str, blocks: list[dict[str, Any]], original_text: str | None = None) -> dict[str, Any]:
    from healthvault.web.extraction import _indicator_label, SIZE_PATTERN, DATE_PATTERN
    from datetime import date
    original = original_text or text
    opcodes = difflib.SequenceMatcher(None, original, text).get_opcodes() if original != text else [('equal', 0, len(text), 0, len(text))]

    def regions(start: int, end: int) -> list[dict[str, Any]]:
        mapped = []
        for tag, a, b, c, d in opcodes:
            if max(start, c) < min(end, d):
                mapped.append((a + max(start-c, 0), a + min(end-c, b-a), False) if tag == 'equal' else (a, b, True))
        found = []
        for index, block in enumerate(blocks):
            a, b = block.get('span_start'), block.get('span_end')
            if not isinstance(a, int) or not isinstance(b, int):
                raw = str(block.get('text', ''))
                if not raw or original.count(raw) != 1:
                    continue
                a = original.index(raw); b = a + len(raw)
            matches = [corrected for left, right, corrected in mapped if max(a, left) < min(b, right)]
            if matches and block.get('polygon'):
                found.append({key: block[key] for key in ('page', 'polygon', 'coordinate_space', 'page_width', 'page_height', 'confidence', 'row_index', 'column_index', 'block_id') if key in block}
                )
                found[-1].update(block_id=block.get('block_id', f'block-{index+1}'), corrected_text=any(matches))
        return found

    date_matches = []
    for match in DATE_PATTERN.finditer(text):
        try:
            if date(*(int(part) for part in match.groups())).isoformat()==preview.get('report',{}).get('report_date'):
                date_matches.append(match)
        except ValueError: pass
    date_field = None
    if date_matches:
        dated = next((match for match in date_matches if re.search(r'报告|检查|采样|report[_ ]date',text[max(0,match.start()-30):match.start()],re.I)),date_matches[0])
        date_field = {'span_start':dated.start(),'span_end':dated.end(),'text':dated.group(),'regions':regions(dated.start(),dated.end())}
        preview['report'].setdefault('evidence',{})['report_date'] = date_field
    for item in [*preview.get('indicators', []), *preview.get('imaging_findings', []), *preview.get('clinical_items', []), *preview.get('other_findings', [])]:
        evidence = item['evidence']; quote = evidence['quote']
        line_start = int(evidence.get('line_start', 1))
        start = sum(len(line) for line in text.splitlines(keepends=True)[:line_start-1])
        position = text.find(quote, start)
        if position < 0:
            match = re.search(r'\r?\n'.join(re.escape(part) for part in quote.split('\n')), text[start:])
            if match is None:
                continue
            position = start + match.start(); quote = match.group()
        end = position + len(quote)
        evidence.update(span_start=position, span_end=end, regions=regions(position, end))
        fields = {}
        def field(name: str, left: int, right: int):
            fields[name] = {'span_start': position+left, 'span_end': position+right,
                            'text': quote[left:right], 'regions': regions(position+left, position+right)}
        if 'clinical_item_id' in item:
            value = item['original_text']
            if quote.count(value) == 1:
                left = quote.index(value); field('original_text', left, left+len(value))
                evidence.update(span_start=position+left, span_end=position+left+len(value), regions=regions(position+left, position+left+len(value)))
            expression = item.get('date_original')
            if expression and quote.count(expression) == 1:
                left = quote.index(expression); field('date_expression', left, left+len(expression))
            located = evidence['regions']
            if located and len({region.get('page', 1) for region in located}) == 1:
                points = [point for region in located for point in region['polygon']]
                x1, x2 = min(point[0] for point in points), max(point[0] for point in points)
                y1, y2 = min(point[1] for point in points), max(point[1] for point in points)
                evidence.update(page=located[0].get('page', 1), polygon=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        elif 'code' in item:
            canonical = re.fullmatch(r'result\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)', quote)
            if canonical:
                for group, name in enumerate(('indicator', 'value', 'unit', 'reference_lower', 'reference_upper'), 1):
                    raw = canonical.group(group)
                    left = canonical.start(group) + len(raw) - len(raw.lstrip())
                    field(name, left, left + len(raw.strip()))
            label = _indicator_label(quote)
            if label:
                field('indicator', 0, label[1])
                numbers = list(re.finditer(r'(?<![\d.])-?\d+(?:\.\d+)?(?![\d.])', quote[label[1]:]))
                if numbers:
                    number = numbers[0]; field('value', label[1]+number.start(), label[1]+number.end())
                range_match = re.search(r'(\d+(?:\.\d+)?)\s*[-~—–至]\s*(\d+(?:\.\d+)?)', quote[label[1]:])
                if range_match:
                    for number, name in ((1, 'reference_lower'), (2, 'reference_upper')):
                        field(name, label[1]+range_match.start(number), label[1]+range_match.end(number))
                unit = re.search(re.escape(item['unit']), quote[label[1]:], re.I)
                if unit:
                    field('unit', label[1]+unit.start(), label[1]+unit.end())
        elif 'finding_type' in item:
            matches = [match for match in SIZE_PATTERN.finditer(quote) if match.group('dimensions').strip() == item['original_dimensions']]
            if len(matches) == 1:
                match = matches[0]
                field('dimensions', match.start('dimensions'), match.end('dimensions'))
                field('unit', match.start('unit'), match.end('unit'))
            for name in ('body_site', 'location'):
                value = item.get(name)
                if value and quote.count(value) == 1:
                    left = quote.index(value); field(name, left, left+len(value))
            for name, tokens in (('laterality',('左','left') if item.get('laterality')=='left' else ('右','right') if item.get('laterality')=='right' else ()),
                                 ('finding_type',('结节','nodule') if item['finding_type']=='nodule' else ('肌瘤','fibroid'))):
                token = next((token for token in tokens if token.casefold() in quote.casefold()),None)
                if token:
                    left = quote.casefold().index(token.casefold()); field(name,left,left+len(token))
            if item.get('body_site') and 'body_site' not in fields:
                value = item['body_site']; left = text.rfind(value, max(0,start-200),position)
                if left>=0:
                    fields['body_site'] = {'span_start':left,'span_end':left+len(value),'text':value,'regions':regions(left,left+len(value)), 'relationship':'nearby_section_header'}
        if date_field and ('code' in item or 'finding_type' in item): fields['report_date'] = date_field
        evidence['fields'] = fields
        if blocks and not evidence['regions']:
            preview.setdefault('warnings', []).append({'code':'field_region_unresolved', 'message':'字段位置未能与原件区域对应，请对照原件核对。', 'quote':quote})
    return preview
