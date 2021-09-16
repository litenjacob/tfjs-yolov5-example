import React from 'react';
import ReactDOM from 'react-dom';
import MagicDropzone from 'react-magic-dropzone';

import './styles.css';
// import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
// setWasmPaths('./tfjs-yolov5-example/');
const tf = require('@tensorflow/tfjs');

const weights = `${process.env.PUBLIC_URL}/web_model/model.json`;

const names = [
  'c',
  'c1',
  'castle',
  'f',
  'f1',
  'g',
  'g1',
  'g2',
  'm',
  'm1',
  'm2',
  'm3',
  's',
  's1',
  's2',
  'w',
  'w1',
];

const [modelWeight, modelHeight] = [416, 416];

class App extends React.Component {
  state = {
    model: null,
    preview: '',
    predictions: [],
  };

  async componentDidMount() {
    // await tf.setBackend('wasm');
    // await tf.ready();
    tf.loadGraphModel(weights).then((model) => {
      console.log({ model });
      this.setState({
        model: model,
      });
    });
  }

  onDrop = (accepted, rejected, links) => {
    this.setState({ preview: accepted[0].preview || links[0] });
  };

  cropToCanvas = (image, canvas, ctx) => {
    const naturalWidth = image.naturalWidth;
    const naturalHeight = image.naturalHeight;

    // canvas.width = image.width;
    // canvas.height = image.height;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const ratio = Math.min(
      canvas.width / image.naturalWidth,
      canvas.height / image.naturalHeight
    );
    const newWidth = Math.round(naturalWidth * ratio);
    const newHeight = Math.round(naturalHeight * ratio);
    ctx.drawImage(
      image,
      0,
      0,
      naturalWidth,
      naturalHeight,
      (canvas.width - newWidth) / 2,
      (canvas.height - newHeight) / 2,
      newWidth,
      newHeight
    );
  };

  onImageChange = (e) => {
    const c = document.getElementById('canvas');
    const ctx = c.getContext('2d');
    this.cropToCanvas(e.target, c, ctx);

    const input = tf.image
      .resizeBilinear(tf.browser.fromPixels(c), [modelWeight, modelHeight])
      .div(255.0)
      .expandDims(0);

    const maxNumBoxes = 49;
    const minScore = 0.5;

    const colorByType = {
      w: 'rgba(13, 112, 239, 0.6)',
      m: 'rgba(59, 64, 61, 0.6)',
      f: 'rgba(70, 86, 50, 0.6)',
      c: 'rgba(222, 191, 57, 0.6)',
      g: 'rgba(130, 188, 68, 0.6)',
      s: 'rgba(134, 126, 92, 0.6)',
      castle: 'rgba(128, 128, 128, 0.6)',
    };

    const before = performance.now();
    this.state.model.executeAsync(input).then((res) => {
      const after = performance.now();
      const div = document.createElement('div');
      div.innerHTML = `Inference took: ${after - before} ms`;
      document.body.appendChild(div);
      // Font options.
      const font = '16px sans-serif';
      ctx.font = font;
      ctx.textBaseline = 'top';

      let [boxes, scores, classes, [validDetectionCount]] = res.map((tensor) =>
        tensor.dataSync()
      );
      boxes = boxes.slice(0, validDetectionCount * 4);
      scores = scores.slice(0, validDetectionCount);
      classes = classes.slice(0, validDetectionCount);

      console.log({ boxes, scores, classes, validDetectionCount });

      // clean the webgl tensors
      input.dispose();
      tf.dispose(res);

      const previousBackend = tf.getBackend();
      tf.setBackend('cpu');
      const indexTensor = tf.tidy(() =>
        tf.image.nonMaxSuppression(
          tf.tensor2d(boxes, [boxes.length / 4, 4]),
          scores,
          maxNumBoxes,
          minScore,
          minScore
        )
      );
      tf.setBackend(previousBackend);

      const indices = indexTensor.dataSync();

      const sortedMatches = Array.from(indices).map((i) => {
        let [x1, y1, x2, y2] = boxes.slice(i * 4, (i + 1) * 4);

        const name = names[classes[i]];
        return {
          type: name.match(/^[a-z]+/)?.[0],
          crowns: Number(name.match(/\d+$/)?.[0] || 0),
          x1,
          y1,
          x2,
          y2,
          xm: (x2 + x1) / 2,
          ym: (y2 + y1) / 2,
          w: x2 - x1,
          h: y2 - y1,
        };
      });

      // FIXME: check castle count
      let map = { 0: {} };
      const castleIndex = sortedMatches.findIndex(
        ({ type }) => type === 'castle'
      );
      if (castleIndex !== -1) {
        const castle = sortedMatches[castleIndex];
        map[0][0] = castle;
        console.log('start', { castle, castleIndex, map });

        const queue = [
          ...sortedMatches.slice(0, castleIndex),
          ...sortedMatches.slice(castleIndex + 1),
        ];

        const getVisitedKey = (x, y) => `${x},${y}`;

        const d = [
          [1, 0],
          [0, 1],
          [-1, 0],
          [0, -1],
        ];
        let visited = {};
        function visit(x, y, lastTile) {
          const visitedKey = getVisitedKey(x, y);
          if (visited[visitedKey]) return;
          visited[visitedKey] = true;

          d.forEach(([dx, dy]) => {
            const [x1s, y1s, x2s, y2s] = [
              lastTile.x1 + dx * lastTile.w,
              lastTile.y1 + dy * lastTile.h,
              lastTile.x2 + dx * lastTile.w,
              lastTile.y2 + dy * lastTile.h,
            ];
            const tileIndex = queue.findIndex(
              ({ xm, ym }) => x1s < xm && x2s > xm && y1s < ym && y2s > ym
            );
            if (tileIndex === -1) return;

            const [newTile] = queue.splice(tileIndex, 1);
            map[y + dy] = map[y + dy] || {};
            map[y + dy][x + dx] = newTile;
            visit(x + dx, y + dy, newTile);
          });
        }
        visit(0, 0, castle);

        const ys = Object.keys(map).map(Number);
        const yMin = Math.min.apply(Math, ys);
        const xs = Object.values(map)
          .map((row) => Object.keys(row).map(Number))
          .flat();
        const xMin = Math.min.apply(Math, xs);

        const normalizedMap = Object.entries(map)
          .map(([yString, row]) => {
            const y = Number(yString) - yMin;
            return [
              y,
              Object.entries(row).reduce((colArray, [xString, tile]) => {
                const x = Number(xString) - xMin;
                tile.x = x;
                tile.y = y;
                colArray[x] = tile;
                return colArray;
              }, []),
            ];
          })
          .reduce((rowArray, [y, row]) => {
            rowArray[y] = row;
            return rowArray;
          }, []);

        const h = normalizedMap.length;
        const w = Math.max.apply(
          Math,
          normalizedMap.map((row) => row.length)
        );

        function getRegion(
          map,
          type,
          x,
          y,
          visited,
          region = { crowns: 0, tiles: [] }
        ) {
          const visitedKey = getVisitedKey(x, y);
          if (!visited[visitedKey] && map[y][x]?.type === type) {
            region.crowns += map[y][x].crowns;
            region.tiles.push(map[y][x]);
            region.type = type;
            visited[visitedKey] = true;

            if (x > 0) getRegion(map, type, x - 1, y, visited, region);
            if (x < w - 1) getRegion(map, type, x + 1, y, visited, region);
            if (y > 0) getRegion(map, type, x, y - 1, visited, region);
            if (y < h - 1) getRegion(map, type, x, y + 1, visited, region);
          }
          return region.tiles.length ? region : false;
        }

        visited = {};
        const regions = normalizedMap
          .flat()
          .map((tile) =>
            getRegion(normalizedMap, tile.type, tile.x, tile.y, visited)
          )
          .filter(Boolean);

        console.log({
          regions,
          normalizedMap,
        });

        regions.forEach((region, regionIndex) => {
          ctx.strokeStyle = colorByType[region.tiles[0].type];
          ctx.fillStyle = colorByType[region.tiles[0].type];
          ctx.lineWidth = 4;

          let path = new Path2D();
          path.moveTo(
            region.tiles[0].x1 * c.width,
            region.tiles[0].y2 * c.height
          );
          let [initialX, initialY, type] = [
            region.tiles[0].x,
            region.tiles[0].y,
            region.tiles[0].type,
          ];
          let [x, y, corner] = [initialX, initialY, 1];

          let i = 0;
          do {
            for (let dCorner = 0; dCorner < 4; dCorner++) {
              i++;
              const { x1, y1, x2, y2 } = normalizedMap[y]?.[x];

              if (corner === 0) {
                path.lineTo(x1 * c.width, y2 * c.height);

                if (region.tiles.includes(normalizedMap[y + 1]?.[x - 1])) {
                  x = x - 1;
                  y = y + 1;
                  corner = 3;
                  continue;
                }

                if (region.tiles.includes(normalizedMap[y]?.[x - 1])) {
                  x = x - 1;
                  corner = 0;
                  continue;
                }
              }
              if (corner === 1) {
                path.lineTo(x1 * c.width, y1 * c.height);

                if (region.tiles.includes(normalizedMap[y - 1]?.[x - 1])) {
                  x = x - 1;
                  y = y - 1;
                  corner = 0;
                  continue;
                }

                if (region.tiles.includes(normalizedMap[y - 1]?.[x])) {
                  y = y - 1;
                  corner = 1;
                  continue;
                }
              }
              if (corner === 2) {
                path.lineTo(x2 * c.width, y1 * c.height);

                if (region.tiles.includes(normalizedMap[y - 1]?.[x + 1])) {
                  x = x + 1;
                  y = y - 1;
                  corner = 1;
                  continue;
                }

                if (region.tiles.includes(normalizedMap[y]?.[x + 1])) {
                  x = x + 1;
                  corner = 2;
                  continue;
                }
              }
              if (corner === 3) {
                path.lineTo(x2 * c.width, y2 * c.height);

                if (region.tiles.includes(normalizedMap[y + 1]?.[x + 1])) {
                  x = x + 1;
                  y = y + 1;
                  corner = 2;
                  continue;
                }

                if (region.tiles.includes(normalizedMap[y + 1]?.[x])) {
                  y = y + 1;
                  corner = 3;
                  continue;
                }
              }
              corner = (corner + 1) % 4;
            }
          } while (
            !(initialX === x && initialY === y && corner === 1) &&
            i < 1000
          );

          path.closePath();
          ctx.fill(path, 'nonzero');

          ctx.shadowColor = 'black';
          ctx.shadowBlur = 3;
          ctx.fillStyle = 'white';
          ctx.textAlign = 'right';
          ctx.textBaseline = 'middle';

          const ch = region.tiles[0].h * c.height;
          const fh = Math.round(ch / 4.5);
          ctx.font = `bold ${fh}px Arial`;

          if (type !== 'castle' && region.crowns !== 0) {
            ctx.fillText(
              `${region.crowns} 👑`,
              c.width * region.tiles[0].x2 - fh / 1.3,
              c.height * region.tiles[0].ym - fh * 1.3
            );

            ctx.fillText(
              `× ${region.tiles.length} ⏺️`,
              c.width * region.tiles[0].x2 - fh / 1.3,
              c.height * region.tiles[0].ym
            );

            ctx.fillText(
              `= ${region.crowns * region.tiles.length} #️⃣`,
              c.width * region.tiles[0].x2 - fh / 1.3,
              c.height * region.tiles[0].ym + fh * 1.3
            );
          }

          ctx.shadowBlur = 0;
        });

        // regions.forEach((region) => {
        //   region.tiles.forEach((tile, tileIndex, tiles) => {
        //     const colorByType = {
        //       w: '#0D70EF',
        //       m: '#3B403D',
        //       f: '#465632',
        //       c: '#DEBF39',
        //       g: '#82BC44',
        //       s: '#867E5C',
        //     };

        //     ctx.strokeStyle = colorByType[tile.type];
        //     ctx.lineWidth = 4;

        //     console.log('regiontile', { region, tile });

        //     [
        //       [1, 0],
        //       [0, 1],
        //       [-1, 0],
        //       [0, -1],
        //     ].forEach(([dx, dy]) => {
        //       console.log(
        //         'cmp',
        //         normalizedMap[tile.y + dy]?.[tile.x + dx]?.type,
        //         tile.type
        //       );
        //       if (
        //         normalizedMap[tile.y + dy]?.[tile.x + dx]?.type !== tile.type
        //       ) {
        //         ctx.beginPath();
        //         ctx.moveTo(
        //           dx == 1 ? tile.x2 * c.width : tile.x1 * c.width,
        //           dy == 1 ? tile.y2 * c.width : tile.y1 * c.width
        //         );
        //         ctx.lineTo(
        //           dx == -1 ? tile.x1 * c.width : tile.x2 * c.width,
        //           dy == -1 ? tile.y1 * c.width : tile.y2 * c.width
        //         );
        //         ctx.stroke();
        //         console.log('strokey', tile, dx, dy);
        //       }
        //     });

        //     // if (tileIndex === tiles.length - 1) ctx.stroke();
        //   });
        // });

        // for (let y = 0; y < h; y++) {
        //   for (let x = 0; x < w; x++) {
        //     const visitedKey = getVisitedKey(x, y);
        //     if (visited[visitedKey]) continue;
        //     visited[visitedKey] = true;

        //     const tile = normalizedMap[y][x];
        //     if (!tile) continue;
        //   }
        // }
      }
      console.log({ castleIndex, map, sortedMatches });
    });
  };

  render() {
    return (
      <div className="Dropzone-page">
        {this.state.model ? (
          <>
            <div>
              {(
                this.state.model.artifacts.weightData.byteLength /
                1024 /
                1024
              ).toFixed(3)}{' '}
              Mb
            </div>
            <MagicDropzone
              className="Dropzone"
              accept="image/jpeg, image/png, .jpg, .jpeg, .png"
              multiple={false}
              onDrop={this.onDrop}
            >
              {this.state.preview ? (
                <img
                  alt="upload preview"
                  onLoad={this.onImageChange}
                  className="Dropzone-img"
                  src={this.state.preview}
                />
              ) : (
                'Choose or drop a file.'
              )}
              <canvas id="canvas" width="640" height="640" />
            </MagicDropzone>
            <div
              style={{
                position: 'relative',
              }}
            >
              {normalizedMap.flat().map((tile) => {
                const tileWidth = 20;
                return (
                  <div
                    style={{
                      width: tileWidth,
                      height: tileWidth,
                      backgroundColor: colorByType[tile.type],
                      position: 'absolute',
                      left: tileWidth * tile.x,
                      top: tileWidth * tile.y,
                    }}
                  />
                );
              })}
            </div>
            ;
          </>
        ) : (
          <div className="Dropzone">Loading model...</div>
        )}
      </div>
    );
  }
}

const rootElement = document.getElementById('root');
ReactDOM.render(<App />, rootElement);
