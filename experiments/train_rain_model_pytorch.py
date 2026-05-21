from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "weather_project_dataset.csv"
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001


class RainPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {DATA_PATH}. "
            "Run `python src/merge_weather_data.py` first."
        )

    df = pd.read_csv(DATA_PATH)

    feature_columns = ["min_temp", "max_temp", "rainfall", "sunshine", "rain_today"]
    X = df[feature_columns]
    y = df["rain_tomorrow"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = RainPredictor()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {average_loss:.4f}")

    model.eval()
    with torch.no_grad():
        test_probabilities = model(X_test_tensor)
        test_predictions = (test_probabilities >= 0.5).int()

    y_true = y_test_tensor.numpy().flatten().astype(int)
    y_pred = test_predictions.numpy().flatten().astype(int)
    accuracy = (y_true == y_pred).mean()

    print()
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Classification report:")
    print(classification_report(y_true, y_pred))

    sample = pd.DataFrame(
        [
            {
                "min_temp": 14.0,
                "max_temp": 24.0,
                "rainfall": 0.0,
                "sunshine": 30.0,
                "rain_today": 0,
            }
        ]
    )
    sample_scaled = scaler.transform(sample)
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

    with torch.no_grad():
        sample_probability = model(sample_tensor).item()

    sample_prediction = 1 if sample_probability >= 0.5 else 0

    print("Sample prediction:")
    print(f"Will it rain tomorrow? {sample_prediction}")
    print(f"Probability of rain tomorrow: {sample_probability:.4f}")


if __name__ == "__main__":
    main()
