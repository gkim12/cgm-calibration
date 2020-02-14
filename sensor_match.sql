SELECT measurements.user_id, measurements.occurred_at, measurements.id, measurements.value, scans.sensor_id
FROM measurements
JOIN scans ON
measurements.scan_id=scans.id
