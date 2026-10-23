"""Domain models for deterministic synthetic health report previews."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class IndicatorSpec:
    display_name: str
    unit: str


INDICATOR_CATALOG = {
    "fasting_glucose": IndicatorSpec("空腹血糖", "mmol/L"),
    "hba1c": IndicatorSpec("糖化血红蛋白", "%"),
    "ldl_cholesterol": IndicatorSpec("低密度脂蛋白胆固醇", "mmol/L"),
    "total_cholesterol": IndicatorSpec("总胆固醇", "mmol/L"),
    "hdl_cholesterol": IndicatorSpec("高密度脂蛋白胆固醇", "mmol/L"),
    "triglycerides": IndicatorSpec("甘油三酯", "mmol/L"),
    "white_blood_cells": IndicatorSpec("白细胞计数", "10^9/L"),
    "hemoglobin": IndicatorSpec("血红蛋白", "g/L"),
    "platelets": IndicatorSpec("血小板计数", "10^9/L"),
    "creatinine": IndicatorSpec("肌酐", "umol/L"),
    "egfr": IndicatorSpec("估算肾小球滤过率", "mL/min/1.73m2"),
    "alt": IndicatorSpec("丙氨酸氨基转移酶", "U/L"),
    "ast": IndicatorSpec("天冬氨酸氨基转移酶", "U/L"),
    "tsh": IndicatorSpec("促甲状腺激素", "mIU/L"),
    "ferritin": IndicatorSpec("铁蛋白", "ng/mL"),
    "systolic_bp": IndicatorSpec("收缩压", "mmHg"),
    "diastolic_bp": IndicatorSpec("舒张压", "mmHg"),
    "heart_rate": IndicatorSpec("心率", "bpm"),
    "body_temperature": IndicatorSpec("体温", "°C"),
    "height": IndicatorSpec("身高", "cm"),
    "weight": IndicatorSpec("体重", "kg"),
    "bmi": IndicatorSpec("体质指数", "kg/m2"),
    "waist_circumference": IndicatorSpec("腰围", "cm"),
    "postprandial_glucose_2h": IndicatorSpec("餐后2小时血糖", "mmol/L"),
    "random_glucose": IndicatorSpec("随机血糖", "mmol/L"),
    "red_blood_cells": IndicatorSpec("红细胞计数", "10^12/L"),
    "hematocrit": IndicatorSpec("红细胞压积", "%"),
    "mcv": IndicatorSpec("平均红细胞体积", "fL"),
    "mch": IndicatorSpec("平均红细胞血红蛋白量", "pg"),
    "mchc": IndicatorSpec("平均红细胞血红蛋白浓度", "g/L"),
    "rdw_cv": IndicatorSpec("红细胞分布宽度变异系数", "%"),
    "neutrophil_percentage": IndicatorSpec("中性粒细胞百分比", "%"),
    "lymphocyte_percentage": IndicatorSpec("淋巴细胞百分比", "%"),
    "monocyte_percentage": IndicatorSpec("单核细胞百分比", "%"),
    "eosinophil_percentage": IndicatorSpec("嗜酸性粒细胞百分比", "%"),
    "basophil_percentage": IndicatorSpec("嗜碱性粒细胞百分比", "%"),
    "neutrophil_count": IndicatorSpec("中性粒细胞计数", "10^9/L"),
    "lymphocyte_count": IndicatorSpec("淋巴细胞计数", "10^9/L"),
    "alkaline_phosphatase": IndicatorSpec("碱性磷酸酶", "U/L"),
    "ggt": IndicatorSpec("γ-谷氨酰转移酶", "U/L"),
    "total_bilirubin": IndicatorSpec("总胆红素", "umol/L"),
    "direct_bilirubin": IndicatorSpec("直接胆红素", "umol/L"),
    "albumin": IndicatorSpec("白蛋白", "g/L"),
    "total_protein": IndicatorSpec("总蛋白", "g/L"),
    "globulin": IndicatorSpec("球蛋白", "g/L"),
    "urea": IndicatorSpec("尿素", "mmol/L"),
    "uric_acid": IndicatorSpec("尿酸", "umol/L"),
    "sodium": IndicatorSpec("钠", "mmol/L"),
    "potassium": IndicatorSpec("钾", "mmol/L"),
    "chloride": IndicatorSpec("氯", "mmol/L"),
    "calcium": IndicatorSpec("钙", "mmol/L"),
    "phosphorus": IndicatorSpec("磷", "mmol/L"),
    "magnesium": IndicatorSpec("镁", "mmol/L"),
    "ft3": IndicatorSpec("游离三碘甲状腺原氨酸", "pmol/L"),
    "ft4": IndicatorSpec("游离甲状腺素", "pmol/L"),
    "serum_iron": IndicatorSpec("血清铁", "umol/L"),
    "transferrin": IndicatorSpec("转铁蛋白", "g/L"),
    "vitamin_d_25oh": IndicatorSpec("25-羟维生素D", "ng/mL"),
    "crp": IndicatorSpec("C反应蛋白", "mg/L"),
    "esr": IndicatorSpec("红细胞沉降率", "mm/h"),
}

INDICATOR_ALIASES = {
    "ldl": "ldl_cholesterol",
    "tc": "total_cholesterol",
    "hdl": "hdl_cholesterol",
    "tg": "triglycerides",
    "wbc": "white_blood_cells",
    "hb": "hemoglobin",
    "plt": "platelets",
    "scr": "creatinine",
    "sbp": "systolic_bp",
    "dbp": "diastolic_bp",
    "hr": "heart_rate",
    "2hpg": "postprandial_glucose_2h",
    "rbc": "red_blood_cells",
    "hct": "hematocrit",
    "neu_pct": "neutrophil_percentage",
    "lym_pct": "lymphocyte_percentage",
    "alp": "alkaline_phosphatase",
    "tbil": "total_bilirubin",
    "dbil": "direct_bilirubin",
    "alb": "albumin",
    "tp": "total_protein",
    "cr": "creatinine",
}

UNIT_ALIASES = {
    "μmol/L": "umol/L",
    "µmol/L": "umol/L",
    "×10^9/L": "10^9/L",
    "10⁹/L": "10^9/L",
    "mL/min/1.73m²": "mL/min/1.73m2",
    "×10^12/L": "10^12/L",
    "10¹²/L": "10^12/L",
    "℃": "°C",
    "kg/m²": "kg/m2",
}


@dataclass(frozen=True)
class ReportSource:
    source_id: str | None
    sha256: str | None


@dataclass(frozen=True)
class ReportMetadata:
    report_id: str | None
    report_date: str | None
    facility: str | None


@dataclass(frozen=True)
class ReferenceRange:
    lower: float
    upper: float


@dataclass(frozen=True)
class Evidence:
    source_id: str | None
    line_start: int
    line_end: int
    quote: str


@dataclass(frozen=True)
class ReportIndicator:
    code: str
    display_name: str
    value: float
    unit: str
    reference_range: ReferenceRange
    flag: str
    evidence: Evidence


@dataclass(frozen=True)
class ClinicalEvidence:
    source_id: str | None
    line_start: int
    line_end: int
    span_start: int
    span_end: int
    quote: str
    page: int | None = None
    polygon: list[list[float]] | None = None


@dataclass(frozen=True)
class ClinicalItemCandidate:
    clinical_item_id: str
    kind: str
    original_text: str
    diagnosis_code: str | None
    normalized_value: str | None
    evidence: ClinicalEvidence
    confidence: float | None
    source_type: str
    recorded_by_provider: bool
    confirmation_status: str
    date_status: str | None
    normalized_date: str | None
    date_original: str | None


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    line: int | None = None
    indicator_code: str | None = None
    expected_unit: str | None = None
    received_unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the stable issue fields and available context."""
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


@dataclass(frozen=True)
class PreviewWarning:
    code: str
    message: str
    line: int
    indicator_code: str
    source_id: str | None
    quote: str
    status: str = "pending_mapping"


@dataclass(frozen=True)
class ReportPreview:
    source: ReportSource
    report: ReportMetadata
    indicators: list[ReportIndicator]
    clinical_items: list[ClinicalItemCandidate]
    issues: list[ValidationIssue]
    warnings: list[PreviewWarning] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the preview using the public domain result shape."""
        has_errors = bool(self.issues)
        return {
            "status": "invalid" if has_errors else "ready",
            "can_confirm": not has_errors,
            "source": asdict(self.source),
            "report": asdict(self.report),
            "indicators": [asdict(indicator) for indicator in self.indicators],
            "clinical_items": [asdict(item) for item in self.clinical_items],
            "issues": [issue.to_dict() for issue in self.issues],
            "warnings": [asdict(warning) for warning in self.warnings],
        }
