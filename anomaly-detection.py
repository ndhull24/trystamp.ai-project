import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


class AdvancedAnomalyDetector:
    """
    Modular anomaly detection component supporting Isolation Forest,
    Local Outlier Factor, and AutoEncoder-based detection.
    """

    def __init__(self, method="isolation_forest", contamination=0.05):
        """
        Initialize the detector.
        Args:
            method: 'isolation_forest', 'lof', or 'autoencoder'
            contamination: expected proportion of anomalies
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()

    def _prepare_data(self, data):
        # Normalize numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        X = data[numeric_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, numeric_cols

    def fit_predict(self, data):
        """
        Train the anomaly detection model and produce anomaly scores.
        Returns:
            DataFrame with anomaly flags and scores.
        """
        X_scaled, numeric_cols = self._prepare_data(data)

        if self.method == "isolation_forest":
            self.model = IsolationForest(contamination=self.contamination, random_state=42)
            preds = self.model.fit_predict(X_scaled)
            scores = self.model.decision_function(X_scaled)

        elif self.method == "lof":
            self.model = LocalOutlierFactor(contamination=self.contamination, novelty=False)
            preds = self.model.fit_predict(X_scaled)
            scores = self.model.negative_outlier_factor_

        elif self.method == "autoencoder":
            n_features = X_scaled.shape[1]
            self.model = Sequential([
                Dense(64, activation='relu', input_dim=n_features),
                Dense(32, activation='relu'),
                Dense(64, activation='relu'),
                Dense(n_features, activation='sigmoid')
            ])
            self.model.compile(optimizer='adam', loss='mse')
            self.model.fit(X_scaled, X_scaled, epochs=30, batch_size=16, verbose=0)

            reconstructions = self.model.predict(X_scaled)
            mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
            threshold = np.percentile(mse, 100 * (1 - self.contamination))
            preds = np.where(mse > threshold, -1, 1)
            scores = mse

        else:
            raise ValueError("Invalid method. Choose 'isolation_forest', 'lof', or 'autoencoder'.")

        K.clear_session()  # clear Keras backend after use
        result = data.copy()
        result["anomaly_flag"] = preds
        result["anomaly_score"] = scores
        anomalies = result[result["anomaly_flag"] == -1]
        return anomalies, result

