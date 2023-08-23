import React from "react";

function AddSong({ songInput, setSongInput, handleAddSong }) {
  return (
    <section id="add-song">
      <h2>Add a Song</h2>
      <textarea
        id="song-input"
        placeholder="Paste a song in ABC notation"
        value={songInput}
        onChange={(e) => setSongInput(e.target.value)}
      />
      <button id="add-song-button" onClick={handleAddSong}>
        Add Song
      </button>
    </section>
  );
}

export default AddSong;
