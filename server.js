const express = require('express');
const SerialPort = require('serialport');
const app = express();
const port = 3000;

app.use(express.json());

const serialPort = new SerialPort('/dev/ttyUSB0', { baudRate: 9600 });

app.post('/send', (req, res) => {
    const data = req.body.data;
    serialPort.write(data, (err) => {
        if (err) {
            return res.status(500).send('Error writing to serial port');
        }
        res.send('Data sent to Arduino');
    });
});

app.listen(port, () => {
    console.log(`Server listening at http://localhost:${port}`);
});
