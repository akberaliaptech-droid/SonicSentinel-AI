"""SQLAlchemy database models for SonicSentinel AI.
"""
from datetime import datetime
import json
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Category(Base):
    """Acoustic category dynamically registered in continuous learning pipeline."""
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), unique=True, nullable=False, index=True)
    severity = Column(String(16), default="MEDIUM", nullable=False)
    python_sample_count = Column(Integer, default=0)
    gtm_sample_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "severity": self.severity,
            "python_sample_count": self.python_sample_count,
            "gtm_sample_count": self.gtm_sample_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active,
        }


class ModelVersion(Base):
    """Model version tracking for active learning transfer fine-tuning runs."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_tag = Column(String(32), unique=True, nullable=False)
    classes_json = Column(Text, nullable=False)
    f1_macro = Column(Float, default=0.0)
    confusion_matrix_json = Column(Text, nullable=True)
    epochs_trained = Column(Integer, default=0)
    weights_path = Column(String(256), nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow)
    is_current = Column(Boolean, default=False)

    def get_classes(self) -> list:
        try:
            return json.loads(self.classes_json)
        except Exception:
            return []

    def get_confusion_matrix(self) -> Any:
        try:
            return json.loads(self.confusion_matrix_json) if self.confusion_matrix_json else None
        except Exception:
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "version_tag": self.version_tag,
            "classes": self.get_classes(),
            "f1_macro": round(self.f1_macro, 4),
            "confusion_matrix": self.get_confusion_matrix(),
            "epochs_trained": self.epochs_trained,
            "weights_path": self.weights_path,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "is_current": self.is_current,
        }


class Incident(Base):
    """Forensic incident record storing dual-model inference, radar DOA, and evidence hashes."""
    __tablename__ = "incidents"

    id = Column(String(64), primary_key=True)  # UUID or INC-YYYYMMDD-XXXX
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    audio_sha256 = Column(String(64), nullable=False, index=True)
    audio_filepath = Column(String(256), nullable=False)

    # Classification & Arbitration
    predicted_category = Column(String(64), nullable=False)
    severity = Column(String(16), nullable=False)
    model_a_class = Column(String(64), nullable=False)
    model_a_confidence = Column(Float, nullable=False)
    model_b_class = Column(String(64), nullable=False)
    model_b_confidence = Column(Float, nullable=False)
    confidence_margin = Column(Float, nullable=False)
    arbitration_status = Column(String(32), nullable=False)
    requires_manual_review = Column(Boolean, default=False, index=True)

    # Signal & Acoustics Forensics
    snr_db = Column(Float, nullable=False)
    rms_amplitude = Column(Float, nullable=False)
    azimuth_deg = Column(Float, nullable=False)
    tdoa_seconds = Column(Float, nullable=False)

    # Evidence artifacts
    heatmap_filepath = Column(String(256), nullable=True)
    waveform_filepath = Column(String(256), nullable=True)
    pdf_report_filepath = Column(String(256), nullable=True)

    # Operator review queue
    review_status = Column(String(32), default="RESOLVED")  # PENDING, APPROVED, OVERRIDDEN
    reviewed_by = Column(String(64), nullable=True)
    review_notes = Column(Text, nullable=True)

    # Ambient Speech & Keyword Surveillance Forensics
    detected_speech_transcript = Column(Text, nullable=True)
    trigger_keyword = Column(String(64), nullable=True, index=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "audio_sha256": self.audio_sha256,
            "predicted_category": self.predicted_category,
            "severity": self.severity,
            "model_a_class": self.model_a_class,
            "model_a_confidence": round(self.model_a_confidence, 4),
            "model_b_class": self.model_b_class,
            "model_b_confidence": round(self.model_b_confidence, 4),
            "confidence_margin": round(self.confidence_margin, 4),
            "arbitration_status": self.arbitration_status,
            "requires_manual_review": self.requires_manual_review,
            "snr_db": round(self.snr_db, 2),
            "rms_amplitude": round(self.rms_amplitude, 5),
            "azimuth_deg": round(self.azimuth_deg, 1),
            "tdoa_seconds": round(self.tdoa_seconds, 7),
            "heatmap_url": f"/api/v1/incidents/{self.id}/heatmap" if self.heatmap_filepath else None,
            "pdf_report_url": f"/api/v1/incidents/export-pdf/{self.id}" if self.pdf_report_filepath else None,
            "review_status": self.review_status,
            "reviewed_by": self.reviewed_by,
            "detected_speech_transcript": self.detected_speech_transcript,
            "trigger_keyword": self.trigger_keyword,
        }


class AuditLog(Base):
    """Cryptographic and administrative audit trail."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String(64), nullable=False)
    actor = Column(String(64), default="system")
    details_json = Column(Text, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "action": self.action,
            "actor": self.actor,
            "details": json.loads(self.details_json) if self.details_json else {},
        }
