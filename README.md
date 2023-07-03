# rnn-jukebox
A node-driven jukebox based on Magenta's performance-rnn, make to run on Horizon-X3 Pi or Raspberry Pi (CPU)

---

`Hardware (X3Pi)`

![](./imgs/photo.jpg)

`Output (MP4)`

<video src="https://raw.githubusercontent.com/luan007/rnn-jukebox/main/imgs/output.mp4" controls style="max-width: 400px;"></video>


----

# How does it work?

The project 

```mermaid
  flowchart TD
      O[Node.js Parent Process] --> A
      A[Node.js + TF.JS] -->|Def Model| B(Magenta's Performance RNN Model)
      -->|Log All Notes +\n Time Signature| O
      O -->|Clock\nSend MIDI CMD On Time| C{Fluid Synth - Socket}
      C -->|Output| D[Audio Output]
```