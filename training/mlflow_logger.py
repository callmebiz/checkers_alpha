"""Lightweight MLflow integration wrapper.

This module attempts to import MLflow and provides simple helpers used by the Trainer.
If MLflow is not installed, the functions are safe no-ops (they log a message).
"""
import os
import logging

logger = logging.getLogger(__name__)

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    _MLFLOW_AVAILABLE = False

def init_mlflow(experiment_name=None, run_name=None, tracking_uri=None, tags=None):
    """
    Initialize MLflow for this process.

    Defaults:
      - tracking DB: `sqlite:///mlflow.db` in project root
      - artifact location: `./mlruns` in project root

    Behavior:
      - Does not create any directories unless MLflow is initialized.
      - If a file-style tracking URI (e.g. 'file:./mlruns' or './mlruns') is
        supplied, it will be mapped to the sqlite backend while using
        `./mlruns` as the artifact location (preserves previous behavior).
    """
    if not _MLFLOW_AVAILABLE:
        logger.info("MLflow not installed; mlflow logging disabled.")
        return None

    effective_tracking_uri = tracking_uri
    artifact_msg = None

    # Default artifact and DB at project root
    default_artifact_location = os.path.join(os.getcwd(), 'mlruns')
    default_db_path = os.path.join(os.getcwd(), 'mlflow.db')

    if not tracking_uri:
        effective_tracking_uri = f"sqlite:///{default_db_path}"
        artifact_msg = f"(using './mlruns' as artifact location)"
    else:
        # Detect file-based URIs like 'file:./mlruns' or plain './mlruns'
        if tracking_uri.startswith('file:') or not (':/' in tracking_uri or tracking_uri.startswith('http')):
            effective_tracking_uri = f"sqlite:///{default_db_path}"
            artifact_msg = f"(mapped file backend '{tracking_uri}' to sqlite backend)"

    try:
        mlflow.set_tracking_uri(effective_tracking_uri)
        if artifact_msg:
            logger.info(f"MLflow tracking URI set to {effective_tracking_uri} {artifact_msg}")
    except Exception:
        logger.warning('Failed to set mlflow tracking URI')

    # Create artifact directory now that MLflow is explicitly enabled
    try:
        os.makedirs(default_artifact_location, exist_ok=True)
    except Exception:
        pass

    # Export tracking URI and artifact root to environment so subprocesses/CLI match
    try:
        os.environ.setdefault('MLFLOW_TRACKING_URI', effective_tracking_uri)
        os.environ.setdefault('MLFLOW_ARTIFACT_ROOT', default_artifact_location)
    except Exception:
        pass

    # Create or set the experiment; when creating a new experiment, set its artifact_location
    if experiment_name:
        try:
            ex = mlflow.get_experiment_by_name(experiment_name)
            if ex is None:
                mlflow.create_experiment(experiment_name, artifact_location=default_artifact_location)
            mlflow.set_experiment(experiment_name)
        except Exception:
            try:
                mlflow.set_experiment(experiment_name)
            except Exception:
                pass

    run = mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(tags)
    logger.info(f"MLflow run started: {run.info.run_id}")
    return run


def log_params(params: dict):
    if not _MLFLOW_AVAILABLE:
        return
    for k, v in params.items():
        try:
            mlflow.log_param(k, v)
        except Exception:
            pass


def log_metrics(metrics: dict, step=None):
    if not _MLFLOW_AVAILABLE:
        return
    for k, v in metrics.items():
        try:
            mlflow.log_metric(k, float(v), step=step)
        except Exception:
            pass


def log_artifact(path, artifact_path=None):
    if not _MLFLOW_AVAILABLE:
        return
    try:
        mlflow.log_artifact(path, artifact_path=artifact_path)
    except Exception:
        pass


def end_run():
    if not _MLFLOW_AVAILABLE:
        return
    try:
        mlflow.end_run()
    except Exception:
        pass
