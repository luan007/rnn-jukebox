/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
import * as tf from '@tensorflow/tfjs-node';
// import "@tensorflow/tfjs-backend-cpu"

// import "@tensorflow/tfjs-node"
let lstmKernel1
let lstmBias1
let lstmKernel2
let lstmBias2
let lstmKernel3
let lstmBias3
let c
let h
let fcB
let fcW
const forgetBias = tf.scalar(1.0);
const activeNotes = new Map();

// How many steps to generate per generateStep call.
// Generating more steps makes it less likely that we'll lag behind in note
// generation. Generating fewer steps makes it less likely that the browser UI
// thread will be starved for cycles.
const STEPS_PER_GENERATE_CALL = 40;
// How much time to try to generate ahead. More time means fewer buffer
// underruns, but also makes the lag from UI change to output larger.
const GENERATION_BUFFER_SECONDS = 20;
// If we're this far behind, reset currentTime time to piano.now().
const MAX_GENERATION_LAG_SECONDS = 1;
// If a note is held longer than this, release it.
const MAX_NOTE_DURATION_SECONDS = 3;

const NOTES_PER_OCTAVE = 12;
const DENSITY_BIN_RANGES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
const PITCH_HISTOGRAM_SIZE = NOTES_PER_OCTAVE;

const RESET_RNN_FREQUENCY_MS = 30000;

let pitchHistogramEncoding
let noteDensityEncoding
let conditioned = false;

let currentPianoTimeSec = 0;
// When the piano roll starts in browser-time via performance.now().
let pianoStartTimestampMs = 0;

let currentVelocity = 100;

const MIN_MIDI_PITCH = 0;
const MAX_MIDI_PITCH = 127;
const VELOCITY_BINS = 32;
const MAX_SHIFT_STEPS = 100;
const STEPS_PER_SECOND = 25;


var noteDensityIdx = 4;
noteDensityEncoding =
    tf.oneHot(tf.tensor1d([noteDensityIdx + 1], 'int32'), DENSITY_BIN_RANGES.length + 1).as1D();


var buffer = tf.buffer([PITCH_HISTOGRAM_SIZE], 'float32');
for (var i = 0; i < PITCH_HISTOGRAM_SIZE; i++) {
    buffer.set(1 / PITCH_HISTOGRAM_SIZE, i);
}
pitchHistogramEncoding = buffer.toTensor();

function getConditioning() {
    return tf.tidy(function () {
        // const size = 1 + (noteDensityEncoding.shape[0]) +
        //     (pitchHistogramEncoding.shape[0]);
        // const conditioning =
        //     tf.oneHot(tf.tensor1d([0], 'int32'), size).as1D();
        // return conditioning;

        var casted = tf.cast(
            noteDensityEncoding, 'float32'
        )
        const axis = 0;
        const conditioningValues =
            casted.concat(pitchHistogramEncoding, axis);
        return tf.tensor1d([0], 'float32').concat(conditioningValues, axis);
    });
}

// The unique id of the currently scheduled setTimeout loop.
let currentLoopId = 0;

const EVENT_RANGES = [
    ['note_on', MIN_MIDI_PITCH, MAX_MIDI_PITCH],
    ['note_off', MIN_MIDI_PITCH, MAX_MIDI_PITCH],
    ['time_shift', 1, MAX_SHIFT_STEPS],
    ['velocity_change', 1, VELOCITY_BINS],
];

function calculateEventSize() {
    let eventOffset = 0;
    for (const eventRange of EVENT_RANGES) {
        const minValue = eventRange[1];
        const maxValue = eventRange[2];
        eventOffset += maxValue - minValue + 1;
    }
    return eventOffset;
}



const EVENT_SIZE = calculateEventSize();
const PRIMER_IDX = 355;  // shift 1s.
let lastSample = tf.scalar(PRIMER_IDX, 'int32');
const CHECKPOINT_URL = 'https://storage.googleapis.com/' +
    'download.magenta.tensorflow.org/models/performance_rnn/tfjs';
start();

import express from 'express';
import servestatic from 'serve-static';
var app = express();
app.use(servestatic("./models"));
app.listen(9988)

let modelReady = false;


function updateConditioningParams() {
    const pitchHistogram = [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    // const pitchHistogram = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    if (noteDensityEncoding != null) {
        noteDensityEncoding.dispose();
        noteDensityEncoding = null;
    }
    const noteDensityIdx = 2;
    const noteDensity = DENSITY_BIN_RANGES[noteDensityIdx];
    // densityDisplay.innerHTML = noteDensity.toString();

    noteDensityEncoding =
        tf.oneHot(
            tf.tensor1d([noteDensityIdx + 1], 'int32'),
            DENSITY_BIN_RANGES.length + 1).as1D();

    if (pitchHistogramEncoding != null) {
        pitchHistogramEncoding.dispose();
        pitchHistogramEncoding = null;
    }
    const buffer = tf.buffer([PITCH_HISTOGRAM_SIZE], 'float32');
    const pitchHistogramTotal = pitchHistogram.reduce((prev, val) => {
        return prev + val;
    });
    for (let i = 0; i < PITCH_HISTOGRAM_SIZE; i++) {
        buffer.set(pitchHistogram[i] / pitchHistogramTotal, i);
    }
    pitchHistogramEncoding = buffer.toTensor();
}

updateConditioningParams();


function start() {
    // console.log("model loading!")
    tf.io.loadWeights([{ "paths": ["group1-shard1of6", "group1-shard2of6", "group1-shard3of6", "group1-shard4of6", "group1-shard5of6", "group1-shard6of6"], "weights": [{ "dtype": "float32", "shape": [921, 2048], "name": "rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel" }, { "dtype": "float32", "shape": [2048], "name": "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias" }, { "dtype": "float32", "shape": [512, 388], "name": "fully_connected/weights" }, { "dtype": "float32", "shape": [1024, 2048], "name": "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel" }, { "dtype": "float32", "shape": [2048], "name": "rnn/multi_rnn_cell/cell_2/basic_lstm_cell/bias" }, { "dtype": "float32", "shape": [388], "name": "fully_connected/biases" }, { "dtype": "float32", "shape": [2048], "name": "rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias" }, { "dtype": "float32", "shape": [1024, 2048], "name": "rnn/multi_rnn_cell/cell_2/basic_lstm_cell/kernel" }] }]
        , "http://localhost:9988")
        .then((vars) => {
            lstmKernel1 =
                vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel']
            lstmBias1 = vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias']

            lstmKernel2 =
                vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel']
            lstmBias2 = vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias']

            lstmKernel3 =
                vars['rnn/multi_rnn_cell/cell_2/basic_lstm_cell/kernel']
            lstmBias3 = vars['rnn/multi_rnn_cell/cell_2/basic_lstm_cell/bias']

            fcB = vars['fully_connected/biases']
            fcW = vars['fully_connected/weights']
            modelReady = true;
            // console.log("model loaded!")
            resetRnn();
        });
}

function resetRnn() {
    c = [
        tf.zeros([1, lstmBias1.shape[0] / 4]),
        tf.zeros([1, lstmBias2.shape[0] / 4]),
        tf.zeros([1, lstmBias3.shape[0] / 4]),
    ];
    h = [
        tf.zeros([1, lstmBias1.shape[0] / 4]),
        tf.zeros([1, lstmBias2.shape[0] / 4]),
        tf.zeros([1, lstmBias3.shape[0] / 4]),
    ];
    if (lastSample != null) {
        lastSample.dispose();
    }
    lastSample = tf.scalar(PRIMER_IDX, 'int32');
    currentPianoTimeSec = Date.now() / 1000
    pianoStartTimestampMs = performance.now() - currentPianoTimeSec * 1000;
    currentLoopId++;
    generateStep(currentLoopId);
}

const notes = ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'b'];

// const pitchHistogramElements = notes.map(
//     note => document.getElementById('pitch-' + note) as HTMLInputElement);
// const histogramDisplayElements = notes.map(
//     note => document.getElementById('hist-' + note) as HTMLDivElement);

let preset1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
let preset2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

async function generateStep(loopId) {
    if (loopId < currentLoopId) {
        // Was part of an outdated generateStep() scheduled via setTimeout.
        return;
    }

    const lstm1 = (data, c, h) =>
        tf.basicLSTMCell(forgetBias, lstmKernel1, lstmBias1, data, c, h);
    const lstm2 = (data, c, h) =>
        tf.basicLSTMCell(forgetBias, lstmKernel2, lstmBias2, data, c, h);
    const lstm3 = (data, c, h) =>
        tf.basicLSTMCell(forgetBias, lstmKernel3, lstmBias3, data, c, h);

    let outputs = [];
    [c, h, outputs] = tf.tidy(() => {
        // Generate some notes.
        const innerOuts = [];
        for (let i = 0; i < STEPS_PER_GENERATE_CALL; i++) {
            // Use last sampled output as the next input.
            const eventInput = tf.oneHot(
                lastSample.as1D(), EVENT_SIZE).as1D();
            // Dispose the last sample from the previous generate call, since we
            // kept it.
            if (i === 0) {
                lastSample.dispose();
            }
            const conditioning = getConditioning();
            const axis = 0;
            const input = conditioning.concat(tf.cast(eventInput, 'float32'), axis).toFloat();
            const output =
                tf.multiRNNCell([lstm1, lstm2, lstm3], input.as2D(1, -1), c, h);
            c.forEach(c => c.dispose());
            h.forEach(h => h.dispose());
            c = output[0];
            h = output[1];

            const outputH = h[2];
            const logits = outputH.matMul(fcW).add(fcB);

            const sampledOutput = tf.multinomial(logits.as1D(), 1).asScalar();

            innerOuts.push(sampledOutput);
            lastSample = sampledOutput;
        }
        return [c, h, innerOuts]
    });

    for (let i = 0; i < outputs.length; i++) {
        parse(outputs[i].dataSync()[0]);
    }
    const delta = Math.max(
        0, currentPianoTimeSec - Date.now() / 1000 - GENERATION_BUFFER_SECONDS);
    setTimeout(() => generateStep(loopId), delta * 1000);
}

// import net from "net";
// var sock = net.connect(9977);

// var queued = [];
function queue(v, time) {
    // queued.push({ v, time })
    console.log(JSON.stringify({ v, time: Math.floor(time * 1000), active: true }))
}

// setInterval(() => {
//     var now = Date.now() / 1000;
//     queued.forEach((q, i) => {
//         if (q.time < now) {
//             sock.write(q.v);
//             q.active = false;
//         }
//     });
//     queued = queued.filter(q => q.active);
// }, 1)

function parse(index) {
    let offset = 0;
    for (const eventRange of EVENT_RANGES) {
        const eventType = eventRange[0];
        const minValue = eventRange[1];
        const maxValue = eventRange[2];
        // console.log(eventType, minValue, maxValue)
        if (offset <= index && index <= offset + maxValue - minValue) {
            if (eventType === 'note_on') {
                const noteNum = index - offset;
                activeNotes.set(noteNum, currentPianoTimeSec);
                // console.log(noteNum, currentPianoTimeSec, currentVelocity);
                // setInterval(() => {

                //0-200

                queue(`noteon 0 ${noteNum} ${Math.floor(currentVelocity * 110)}\n`, currentPianoTimeSec)
                // sock.write()
                // }, 1000)
                // return piano.keyDown(
                // noteNum, currentPianoTimeSec, currentVelocity * globalGain / 100);
                return;
            } else if (eventType === 'note_off') {
                const noteNum = index - offset;

                const activeNoteEndTimeSec = activeNotes.get(noteNum);
                // If the note off event is generated for a note that hasn't been
                // pressed, just ignore it.
                if (activeNoteEndTimeSec == null) {
                    return;
                }
                const timeSec = Math.max(currentPianoTimeSec, activeNoteEndTimeSec + .5);
                // piano.keyUp(noteNum, timeSec);

                queue(`noteoff 0 ${noteNum}\n`, timeSec)
                // sock.write(`noteoff 0 ${noteNum}\n`)

                activeNotes.delete(noteNum);
                return;
            } else if (eventType === 'time_shift') {
                currentPianoTimeSec += (index - offset + 1) / STEPS_PER_SECOND;
                activeNotes.forEach((timeSec, noteNum) => {
                    if (currentPianoTimeSec - timeSec > MAX_NOTE_DURATION_SECONDS) {
                        // console.info(
                        //     `Note ${noteNum} has been active for ${currentPianoTimeSec - timeSec}, ` +
                        //     `seconds which is over ${MAX_NOTE_DURATION_SECONDS}, will ` +
                        //     `release.`);
                        // piano.keyUp(noteNum, currentPianoTimeSec);

                        queue(`noteoff 0 ${noteNum}\n`, currentPianoTimeSec)
                        activeNotes.delete(noteNum);
                    }
                });
                return currentPianoTimeSec;
            } else if (eventType === 'velocity_change') {
                currentVelocity = (index - offset + 1) * Math.ceil(127 / VELOCITY_BINS);
                currentVelocity = currentVelocity / 127;
                return currentVelocity;
            } else {
                throw new Error('Could not decode eventType: ' + eventType);
            }
        }
        offset += maxValue - minValue + 1;
    }
    throw new Error(`Could not decode index: ${index}`);
}

// Reset the RNN repeatedly so it doesn't trail off into incoherent musical
// babble.
function resetRnnRepeatedly() {
    if (modelReady) {
        resetRnn();
    }
    setTimeout(resetRnnRepeatedly, RESET_RNN_FREQUENCY_MS);
}
setTimeout(resetRnnRepeatedly, RESET_RNN_FREQUENCY_MS);