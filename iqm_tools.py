"""Small helpers for querying live IQM hardware calibration state.

Self-contained (only depends on iqm-client/iqm-pulla/pandas), no dependency on any
private codebase.
"""
import pandas as pd
from qiskit import QuantumCircuit

from iqm.qiskit_iqm import IQMProvider
from iqm.pulla.pulla import Pulla
from iqm.pulla.utils_qiskit import get_qiskit_compiler
from iqm.iqm_client import IQMClient

_SORT_COLUMNS = {
    "qubit":           None,               # special: sort by QB number
    "frequency":       "frequency_GHz",
    "anharmonicity":   "anharmonicity_MHz",
    "T1":              "T1_us",
    "T2":              "T2_ramsey_us",
}


def get_qubit_params(iqm_url: str, sort_by: str = "qubit", ascending: bool = True) -> pd.DataFrame:
    """Return a DataFrame of live qubit calibration parameters for every qubit on the chip.

    sort_by: one of 'qubit', 'frequency', 'anharmonicity', 'T1', 'T2'
    """
    pulla_    = Pulla(iqm_url)
    provider_ = IQMProvider(iqm_url)
    backend_  = provider_.get_backend()

    compiler = get_qiskit_compiler(pulla_, backend_)
    settings = compiler.get_settings(circuits=[QuantumCircuit(1, 1)])
    qubits   = compiler.chip_topology.qubits

    client   = IQMClient(iqm_url)
    cal_set  = client.get_calibration_set()
    metrics  = client.get_calibration_quality_metrics(calibration_set_id=cal_set.observation_set_id)
    t1_dict, t2_dict = metrics.get_coherence_times(qubits)

    def safe(val):
        return float(val) if val is not None else None

    qubit_params = {}
    for qb in qubits:
        ctrl  = getattr(settings.controllers, qb, None)
        prx12 = getattr(getattr(settings.gates.prx_12.modulated_drag_crf, qb, None), "frequency", None)
        qubit_params[qb] = {
            "frequency_GHz":     safe(ctrl.drive.frequency.value / 1e9) if ctrl  else None,
            "anharmonicity_MHz": safe(prx12.value / 1e6)                if prx12 else None,
            "T1_us":             safe(t1_dict.get(qb) * 1e6)            if t1_dict.get(qb) else None,
            "T2_ramsey_us":      safe(t2_dict.get(qb) * 1e6)            if t2_dict.get(qb) else None,
        }

    df = pd.DataFrame(qubit_params).T.reset_index(names="qubit")
    df.attrs["calibration_set_id"] = str(cal_set.observation_set_id)

    return sort_qubit_params(df, sort_by=sort_by, ascending=ascending)


def sort_qubit_params(df: pd.DataFrame, sort_by: str = "qubit", ascending: bool = True) -> pd.DataFrame:
    col = _SORT_COLUMNS.get(sort_by)
    if col is None:
        df = df.copy()
        df["_n"] = df["qubit"].str.extract(r'(\d+)')[0].astype(int)
        df = df.sort_values("_n", ascending=ascending).drop(columns="_n")
        return df.reset_index(drop=True)
    return df.sort_values(col, ascending=ascending).reset_index(drop=True)
