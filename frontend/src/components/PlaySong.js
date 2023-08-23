import React from "react";

function PlaySong() {
  return (
    <section id="generated-song">
      <h2>Generated Song</h2>
      <pre id="song-notation"></pre>
      <audio id="song-audio" controls></audio>
    </section>
  );
}

export default PlaySong;
