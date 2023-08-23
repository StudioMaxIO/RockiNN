import React from "react";

function GenerateSong({
  seedText,
  setSeedText,
  generationLength,
  setGenerationLength,
  handleGenerateSong
}) {
  return (
    <div>
      <input
        type="text"
        id="seed-input"
        placeholder="Enter seed text"
        value={seedText}
        onChange={(e) => setSeedText(e.target.value)}
      />
      &nbsp;
      <input
        type="text"
        id="gen
        eration-length-input"
        placeholder="Enter generation length"
        value={generationLength}
        onChange={(e) => setGenerationLength(e.target.value)}
      />
      <button id="generate-song-button" onClick={handleGenerateSong}>
        Generate Song
      </button>
    </div>
  );
}

export default GenerateSong;
