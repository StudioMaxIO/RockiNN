from flask import Flask, request, jsonify
import tensorflow as tf
from tqdm import tqdm
import os
import model as mdl  # Assuming the model code is in a file named model.py
import json

app = Flask(__name__)


@app.route('/add-song', methods=['POST'])
def add_song():
    # Logic to add a song to the training set
    return jsonify(status="success")


@app.route('/set-hyperparameters', methods=['POST'])
def set_hyperparameters():
    # Logic to update hyperparameters
    # get hyperparameters from request and save to JSON file in ./data
    hyperparameters = request.get_json()
    with open(os.path.join("data", "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f, indent=2, sort_keys=True)

    return jsonify(status="success")


@app.route('/train-model', methods=['POST'])
def train_model():
    # Logic to train the model

    # get hyperparameters from request
    # hyperparameters = request.get_json()
    # embedding_dim = hyperparameters['embedding_dim']
    # rnn_units = hyperparameters['rnn_units']
    # batch_size = hyperparameters['batch_size']
    # learning_rate = hyperparameters['learning_rate']
    # num_training_iterations = hyperparameters['num_training_iterations']
    # seq_length = hyperparameters['seq_length']
    # optimizer_type = hyperparameters['optimizer_type']
    # checkpoint_dir = hyperparameters['checkpoint_dir']
    # checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

    # get paramaters from ./data/hyperparameters.json
    hyperparameters = {}
    print("loading hyperparameters")
    with open(os.path.join("data", "hyperparameters.json"), "r") as f:
        hyperparameters = json.load(f)
    embedding_dim = hyperparameters['embedding_dim']
    rnn_units = hyperparameters['rnn_units']
    batch_size = hyperparameters['batch_size']
    learning_rate = hyperparameters['learning_rate']
    num_training_iterations = hyperparameters['num_training_iterations']
    seq_length = hyperparameters['seq_length']
    optimizer_type = hyperparameters['optimizer_type']
    checkpoint_dir = hyperparameters['checkpoint_dir']
    checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

    print("preparing data")
    # 1. prepare data - vocab, vectorized_songs = prepare_data()
    vocab, vectorized_songs = mdl.prepare_data()
    vocab_size = len(vocab)

    print("building model")
    # 2. build model (model.py) - vocab size, embedding dim, rnn units, batch size
    model = mdl.build_model(vocab_size, embedding_dim, rnn_units, batch_size)

    print("instantiate optimizer")
    # 3. instantiate optimizer w/ learning rate & optimizer type (model.py)
    optimizer = mdl.instantiate_optimizer(learning_rate, optimizer="Adam")

    print("training model")
    # 4. Begin training
    history = []
    # plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()  # clear if it exists

    for iter in tqdm(range(num_training_iterations)):
        # Grab a batch and propagate it through the network
        x_batch, y_batch = mdl.get_batch(
            vectorized_songs, seq_length, batch_size)
        loss = mdl.train_step(model, optimizer=optimizer, x=x_batch, y=y_batch)

        # Update the progress bar
        history.append(loss.numpy().mean())
        # plotter.plot(history)

        # Update the model with the changed weights!
        if iter % 100 == 0:
            model.save_weights(checkpoint_prefix)

        # Save the trained model and the weights
    model.save_weights(checkpoint_prefix)
    return jsonify(status="success")


@app.route('/generate-song', methods=['POST'])
def generate_song():
    # Logic to generate a song
    vocab, _ = mdl.prepare_data()
    vocab_size = len(vocab)

    # 1. Get hyperparameters from request
    requestParams = request.get_json()
    seed_text = requestParams['seed_text']
    generation_length = requestParams['generation_length']

    # 1. Get hyperparameters from ./data/hyperparameters.json
    hyperparameters = {}
    print("loading hyperparameters")
    with open(os.path.join("data", "hyperparameters.json"), "r") as f:
        hyperparameters = json.load(f)
    embedding_dim = hyperparameters['embedding_dim']
    rnn_units = hyperparameters['rnn_units']
    checkpoint_dir = hyperparameters['checkpoint_dir']

    # 2. Rebuild model with batch size 1
    model = mdl.build_model(
        vocab_size, embedding_dim, rnn_units, batch_size=1)

    # 3. Restore the model weights for the last checkpoint after training
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    # 4. Generate song(s) using the model
    generated_text = mdl.generate_text(model, vocab=vocab, start_string=seed_text,
                                       generation_length=generation_length)
    generated_songs = mdl.extract_song_snippet(generated_text)

    # 5. Return the generated song(s) as a json response
    status = "success" if len(generated_songs) > 0 else "failure"
    return jsonify(status=status, songs=generated_songs)


@app.route('/get-audio', methods=['GET'])
def get_audio():
    # get song from json request
    requestParams = request.get_json()
    song = requestParams['song']
    # Logic to convert a song to audio format
    print(f"""{song}""")
    return jsonify(status="success")


if __name__ == '__main__':
    app.run(debug=True)
