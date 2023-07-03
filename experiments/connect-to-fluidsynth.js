// var childprocess = require("child_process");
// var spa = childprocess.spawn("fluidsynth", `-i -s -o "shell.port=9988" ./SalC5Light2.sf2`.split(" "));
var net = require("net");
var sock = net.connect(9977);
setInterval(() => {
    sock.write("noteon 0 60 100\n")
}, 1000)