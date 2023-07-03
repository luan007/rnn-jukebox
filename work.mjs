
import net from "net";
var sock = net.connect(9977); //, '192.168.50.8');

import childprocess from "child_process";
var sp = childprocess.spawn("node", ["main.mjs"]);

var line = "";
sp.stdout.on('data', (data) => {
    //read line
    data = data.toString();
    for (var i = 0; i < data.length; i++) {
        line += data[i];
        if (data[i] == '\n') {
            try {
                var j = JSON.parse(line);
                queue(
                    j.v, j.time
                )
            }
            catch (e) {

            }
            line = '';
        }
    }
});

var queued = [];
function queue(v, time) {
    queued.push({ v, time, active: true })
    console.log(JSON.stringify({ v, time: Math.floor(time * 1000), active: true }))
}

setInterval(() => {
    var now = Date.now();
    queued.forEach((q, i) => {
        if (q.time < now) {
            sock.write(q.v);
            q.active = false;
        }
    });
    queued = queued.filter(q => q.active);
}, 1)

