import React from "react";

function Hyperparameter({ hyperparameters, setHyperparameters }) {
  const handleChange = (name, value) => {
    setHyperparameters({
      ...hyperparameters,
      [name]: value
    });
  };

  return (
    <section id="hyperparameters">
      <h2>Hyperparameters</h2>
      <label>Number of Training Iterations:</label>
      <input
        type="number"
        value={hyperparameters.num_training_iterations}
        onChange={(e) =>
          handleChange("num_training_iterations", e.target.value)
        }
      />
      <label>Batch Size:</label>
      <input
        type="number"
        value={hyperparameters.batch_size}
        onChange={(e) => handleChange("batch_size", e.target.value)}
      />
      <label>Sequence Length:</label>
      <input
        type="number"
        value={hyperparameters.seq_length}
        onChange={(e) => handleChange("seq_length", e.target.value)}
      />
      <label>Learning Rate:</label>
      <input
        type="number"
        step="0.0001"
        value={hyperparameters.learning_rate}
        onChange={(e) => handleChange("learning_rate", e.target.value)}
      />
      <label>Embedding Dimension:</label>
      <input
        type="number"
        value={hyperparameters.embedding_dim}
        onChange={(e) => handleChange("embedding_dim", e.target.value)}
      />
      <label>RNN Units:</label>
      <input
        type="number"
        value={hyperparameters.rnn_units}
        onChange={(e) => handleChange("rnn_units", e.target.value)}
      />
      <label>Checkpoint Directory:</label>
      <input
        type="text"
        value={hyperparameters.checkpoint_dir}
        onChange={(e) => handleChange("checkpoint_dir", e.target.value)}
      />
    </section>
  );
}

export default Hyperparameter;
