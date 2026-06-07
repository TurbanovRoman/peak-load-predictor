# =============================================================================
# Конфигурация системы прогнозирования пиковых нагрузок (ВКР)
# Стек мониторинга: github.com/artemonsh/grafana-deploy
# =============================================================================

# --- Prometheus ---
PROMETHEUS_URL = "http://localhost:9090"
PROMETHEUS_ACCESS_TOKEN = ""        # Bearer-токен (если требуется)

# --- PromQL-запросы для метрик бэкенда grafana-deploy ---
# Бэкенд: Litestar, метрика: litestar_requests_total{method,path,status_code}
# rate() даёт скорость запросов/сек; sum() агрегирует по всем меткам.
# Окно rate должно быть ≥ step (рекомендуется step * 2..3).
PROMETHEUS_QUERY_RPS = "sum(rate(litestar_requests_total[5m]))"
PROMETHEUS_QUERY_LATENCY_P99 = (
    "histogram_quantile(0.99, "
    "sum(rate(litestar_request_duration_seconds_bucket[5m])) by (le))"
)
PROMETHEUS_QUERY_ACTIVE = "sum(litestar_requests_in_progress)"

# Псевдоним для обратной совместимости с main.py
PROMETHEUS_QUERY = PROMETHEUS_QUERY_RPS

# Параметры выгрузки исторических данных
# step="5min": 1 точка каждые 5 минут = 288 точек/день = 2016 за неделю
# Prometheus scrape_interval=3s, rate-окно 5m — хорошее соответствие.
DEFAULT_STEP = "5min"
DEFAULT_DAYS_AGO = 7               # глубина истории для обучения (дней)

# --- Обучение ---
MODEL_SAVE_DIR = "./saved_models"

# Горизонты прогнозирования (в числе периодов при DEFAULT_STEP="5min"):
#   15 мин = 3 периода, 30 мин = 6, 60 мин = 12
FORECAST_HORIZONS_PERIODS = [3, 6, 12]

# Разбиение данных (в периодах, при DEFAULT_STEP="5min"):
#   24ч * 12 периодов/ч = 288 периодов
SPLIT_TEST_PERIODS = 288           # 24 часа тестовых данных
SPLIT_VAL_PERIODS = 288            # 24 часа валидационных данных

# Для обратной совместимости (используются в main.py как hours, но фактически — периоды)
SPLIT_TEST_HOURS = SPLIT_TEST_PERIODS
SPLIT_VAL_HOURS = SPLIT_VAL_PERIODS

# XGBoost гиперпараметры
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_EARLY_STOPPING = 20

# --- Детекция пиков (глава 2.7.1 ВКР) ---
PEAK_METHOD = "rolling_std"         # "rolling_std" | "percentile"
PEAK_K = 2.0                        # коэффициент σ при rolling_std
PEAK_PERCENTILE = 95.0              # квантиль при методе percentile
PEAK_WINDOW_HOURS = 24              # окно для rolling_std (часов → пересчитывается в периоды)

# Пороги уровней алертов (доля от threshold)
ALERT_WARNING_RATIO = 0.70          # ≥70% от порога → warning
ALERT_CRITICAL_RATIO = 0.85         # ≥85% от порога → critical

# --- Масштабирование ---
# При 5-минутной детализации и типичном трафике grafana-deploy:
TARGET_LOAD_PER_REPLICA = 10.0      # RPS на реплику (бэкенд маленький, 10 RPS на инстанс)
MIN_REPLICAS = 1
MAX_REPLICAS = 10
SCALE_DOWN_BACKOFF = 300            # задержка уменьшения реплик (сек) = 5 минут

# --- Порог переобучения (глава 2.6.4 ВКР) ---
MAPE_RETRAIN_THRESHOLD = 20.0       # при MAPE > 20% инициировать переобучение

# --- Детектор дрейфа ADWIN ---
ADWIN_DELTA        = 0.002  # чувствительность ADWIN: меньше → раньше срабатывает
ADWIN_MIN_OBS      = 30     # минимум наблюдений перед первой проверкой (прогрев)
ADWIN_COOLDOWN     = 20     # шагов паузы после ретрейна (защита от частых срабатываний)
ADWIN_CONFIRMATION = 10     # шагов подтверждения дрейфа перед ретрейном

# --- Принудительный периодический ретрейн ---
RETRAIN_N_FRESH = 0         # ретрейн каждые N шагов независимо от дрейфа (0 = отключено)

# --- Эффект ореола вокруг праздников ---
# Зона «до» и «после» праздника, в которой нагрузка нетипично высока.
# Тьюки не срезает пики внутри этого окна; аналитик видит флаг is_halo для ручного аудита.
HALO_BEFORE = 3   # дней ДО праздника
HALO_AFTER  = 2   # дней ПОСЛЕ праздника

# --- Экзогенные признаки (известны заранее: праздники, акции, кампании) ---
# Используются в NeuralProphet как future_regressor, в XGBoost/LSTM как обычные признаки.
EXOG_COLS = ["is_holiday", "is_halo", "is_campaign", "is_promo"]

# --- Синтетические данные (для демо без Prometheus) ---
SYNTHETIC_DAYS = 30
SYNTHETIC_FREQ = "5min"             # шаг синтетики = шагу реального сбора
SYNTHETIC_BASE_RPS = 5.0            # RPS типичен для небольшого тестового стека
SYNTHETIC_NOISE_STD = 1.0
SYNTHETIC_PEAK_PROB = 0.015
SYNTHETIC_PEAK_MULTIPLIER = 4.0
