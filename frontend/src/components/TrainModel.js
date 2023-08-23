import React from "react";

function TrainModel({ handleTrainModel }) {
  return (
    <button id="train-button" onClick={handleTrainModel}>
      Train Model
    </button>
  );
}

export default TrainModel;
