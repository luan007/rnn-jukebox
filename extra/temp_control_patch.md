## Installing the Fan

patch: ssd1306_i2c.c

set 

```c
#define rotation 2 // 0: 0 degree, 1: 90 degree, 2: 180 degree, 3: 270 degree
```

```bash
mkdir -p /home/sunrise/temp_control
cp temp_control /home/sunrise/temp_control
cp start.sh /home/sunrise/temp_control
```


also

modify temp_control.c to change the fan speed according to the temperature