# celery_worker.py
import os
from celery import Celery

# Configura il broker (Redis) e il backend per il risultato.
broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery('training_tasks', broker=broker_url, backend=result_backend)

# Puoi specificare delle route, se vuoi assegnare i task a queue specifiche.
celery_app.conf.task_routes = {
    'tasks.train_model_task': {'queue': 'training_queue'},
}

if __name__ == "__main__":
    celery_app.start()
