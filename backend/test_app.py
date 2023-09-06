import requests
import json

# Base URL of the Flask app (update with the actual URL if different)
BASE_URL = "http://127.0.0.1:5000"

# Test adding a song


def test_add_song():
    response = requests.post(f"{BASE_URL}/add-song",
                             json={"song": "example_song"})
    print(response.json())


def test_set_hyperparameters():
    print("test setting hyperparameters")
    hyperparameters = {
        "embedding_dim": 256,
        "rnn_units": 1024,
        "batch_size": 4,
        "learning_rate": 5e-3,
        "num_training_iterations": 2000,
        "seq_length": 100,
        "optimizer_type": "Adam",
        "checkpoint_dir": "./training_checkpoints",
    }
    response = requests.post(
        f"{BASE_URL}/set-hyperparameters", json=hyperparameters)
    print(response.json())


def test_train_model():
    print("test training model")
    response = requests.post(f"{BASE_URL}/train-model")
    print(response.json())


def test_generate_song():
    generation_params = {
        "seed_text": "X",
        "generation_length": 1500,
    }
    response = requests.post(
        f"{BASE_URL}/generate-song", json=generation_params)
    # print(response.json())
    return response.json()

# Test getting audio


def test_get_audio(song):
    response = requests.get(f"{BASE_URL}/get-audio", json={"song": song})
    print(response.json())


# Run the tests
if __name__ == "__main__":
    # test_set_hyperparameters()
    # test_train_model()
    # response = test_generate_song()
    # test_get_audio(response['songs'][0])
    test_add_song()
