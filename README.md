# tfjs-yolov5-example

## Usage
Export TensorFlow.js model as described in https://github.com/ultralytics/yolov5/pull/1127

## Local Test
After exported the tfjs model, clone this repo:
```
git clone https://github.com/zldrobit/tfjs-yolov5-example.git
cd tfjs-yolov5-example
```
Install packages with npm:
```
npm install
```
Link YOLOv5 weights directory into public directory:
```
ln -s ../../yolov5/weights/web_model public/web_model
```
Run YOLOv5 detection web service with:
```
npm start
```

## Github Pages Depolyment
Edit `homepage` field in `package.json` changing 
```
"homepage": "https://zldrobit.github.io/tfjs-yolov5-example",
``` 
to 
```
"homepage": "https://GITHUB_USERNAME.github.io/tfjs-yolov5-example",
```


Run
```
npm run deploy
```

PS: This repo assumes the model input resolution is 320x320.
If you change the `--img` value in exporting `*.pb`, change `modelWidth` and `modelHeight` in `src/index.js` accordingly.

EDIT: 
- Add github pages deployment support
- Use GitHub project site (previously use GitHub User Page)

## Reference
https://medium.com/hackernoon/tensorflow-js-real-time-object-detection-in-10-lines-of-code-baf15dfb95b2
