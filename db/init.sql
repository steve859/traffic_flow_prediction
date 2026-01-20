CREATE TABLE IF NOT EXISTS traffic_predictions (
  id BIGSERIAL PRIMARY KEY,
  window_end TIMESTAMPTZ NOT NULL,
  pred_time TIMESTAMPTZ NOT NULL,
  camera_id TEXT NOT NULL,
  pred_flow DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_traffic_predictions_pred_time ON traffic_predictions(pred_time);
CREATE INDEX IF NOT EXISTS idx_traffic_predictions_camera_time ON traffic_predictions(camera_id, pred_time);
