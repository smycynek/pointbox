import { createSignal, For, JSX, onMount, Show, type Component } from 'solid-js';
import styles from './App.module.css';
import { createMutable } from 'solid-js/store';
import { Point } from './Point';
import { groupPointsFromArray } from './tensorGrouping';
import { Color } from './color';
import { functionDemo, getMousePos, enableBackEnd, logMemory, randomSeedCentroid } from './utility';
import { AboutBox } from './AboutBox';
import { Logger } from './Logger';

const App: Component = () => {
  let canvas: HTMLCanvasElement;
  let context: CanvasRenderingContext2D;

  // Canvas height
  const [height, setHeight] = createSignal(0);

  // Text display for centroids
  const [centroid1, setCentroid1] = createSignal('-');
  const [centroid2, setCentroid2] = createSignal('-');

  // Button prompt text
  const [prompt, setPrompt] = createSignal('Group');
  const [maxReached, setMaxReached] = createSignal(false);

  const [isAutoGroupChecked, setIsAutoGroupChecked] = createSignal(false);
  const [engine, setEngine] = createSignal('-'); // backend display for debugging

  let currentCentroid1: Point;
  let currentCentroid2: Point;

  // All point data
  const ptData: Point[] = [];
  const pointStorage = createMutable({
    pointsArray: ptData,
    get points() {
      return this.pointsArray;
    },
    set points(value) {
      this.pointsArray = value;
    },
  });

  const findGroups = async () => {
    if (pointStorage.points.length < 2) {
      setPrompt('Add more points...');
      return;
    }

    if (!currentCentroid1) {
      currentCentroid1 = randomSeedCentroid(height());
    }
    if (!currentCentroid2) {
      currentCentroid2 = randomSeedCentroid(height());
    }
    const data = await groupPointsFromArray(pointStorage.points, [
      currentCentroid1,
      currentCentroid2,
    ]);
    if (!data.group1IsDefault() || !data.group2IsDefault()) {
      currentCentroid1 = data.centerPoints[0];
      currentCentroid2 = data.centerPoints[1];

      const g1Count = data.assignments.filter((a) => a === 0).length;
      const g2Count = data.assignments.length - g1Count;

      if (!data.group1IsDefault()) {
        setCentroid1(`Center 1: ${currentCentroid1} Total: ${g1Count.toString().padStart(3, '0')}`);
      }
      if (!data.group2IsDefault()) {
        setCentroid2(`Center 2: ${currentCentroid2} Total: ${g2Count.toString().padStart(3, '0')}`);
      }

      drawGroups(
        pointStorage.points,
        data.assignments,
        data.centerPoints,
        data.group1IsDefault(),
        data.group2IsDefault(),
        g1Count,
        g2Count
      );
      setPrompt('Regroup');
    } else {
      setPrompt('Add more points...');
    }
  };

  // Only called once to give a reasonable sized canvas that is square
  // and a multiple of 50px
  const resizeCanvas = () => {
    const reducedHeight = window.innerHeight * 0.6;
    const roundedHeight = reducedHeight - (reducedHeight % 50);
    const reducedWidth = window.innerWidth * 0.85;
    const roundedWidth = reducedWidth - (reducedWidth % 50);

    if (roundedWidth > roundedHeight) {
      canvas.width = roundedHeight;
      canvas.height = roundedHeight;
    } else {
      canvas.width = roundedWidth;
      canvas.height = roundedWidth;
    }
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    context = canvas.getContext('2d')!;
    setHeight(canvas.height);
  };

  // 50px spaced grid point for reference
  const drawGrid = () => {
    for (let idx = 50; idx < canvas.width; idx += 50) {
      for (let idy = 50; idy < canvas.height; idy += 50) {
        drawGridPoint(idx, idy);
      }
    }
  };

  const toggleLog = () => {
    console.log('Logging toggle');
    Logger.enabled = !Logger.enabled;
  };

  const init = () => {
    if (context) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      pointStorage.pointsArray = [];
    }
    setMaxReached(false);
    setCentroid1('-');
    setCentroid2('-');
    setPrompt('Group');
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    canvas = document.getElementById('main-canvas')! as HTMLCanvasElement;
    resizeCanvas();
    drawGrid();
  };

  const drawText = (x: number, y: number, text: string) => {
    if (!canvas) {
      init();
    }
    context.strokeStyle = Color.black;
    context.fillStyle = Color.black;
    context.font = '18px monospace';
    context.fillText(text, x, y);
  };

  const drawGridPoint = (x: number, y: number) => {
    drawPoint(x, y, Color.black, 0.75);
  };

  const drawSelectionPoint = (x: number, y: number) => {
    drawPoint(x, y, Color.black, 2);
  };

  const drawGroupPoint = (x: number, y: number, color: Color) => {
    drawPoint(x, y, color, 4);
  };

  const drawCentroid = (x: number, y: number, color: Color, count: number) => {
    if (!canvas) {
      init();
    }
    context.strokeStyle = color;
    context.fillStyle = Color.grey;
    context.lineWidth = 2;
    context.lineDashOffset = 4;
    context.setLineDash([6, 6]);
    context.beginPath();
    context.arc(x, y, 20, 0, 2 * Math.PI);
    context.fill();
    context.stroke();
    drawText(x - 15, y + 6, count.toString().padStart(3, '0'));
  };

  const drawPoint = (
    x: number, // note, screen coords
    y: number,
    color: Color,
    radius: number = 2
  ) => {
    if (!canvas) {
      init();
    }
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = 1;
    context.beginPath();
    context.arc(x, y, radius, 0, 2 * Math.PI);
    context.fill();
    context.stroke();
  };

  const autoGroupChangedHandler: JSX.ChangeEventHandlerUnion<HTMLInputElement, Event> = (
    event: Event
  ) => {
    const target = event.target as HTMLInputElement;
    if (target) {
      setIsAutoGroupChecked(target.checked);
      if (target.checked) {
        onGroup();
      }
    }
  };

  const mouseDownHandler = (data: MouseEvent) => {
    if (!context) {
      init();
    }
    if (pointStorage.points.length > height()) {
      // The max could be much higher, but capping at 200-400 depending on screen size
      setMaxReached(true);
      setPrompt('Max points reached');
      return;
    }
    const pos: Point = getMousePos(canvas, data);
    const roundedX = Math.round(pos.x);
    const roundedY = Math.round(pos.y);
    drawSelectionPoint(roundedX, roundedY); // cartesian to screen

    pointStorage.points.push(new Point(roundedX, height() - roundedY)); // convert from screen to cartesian

    if (isAutoGroupChecked()) {
      onGroup();
    }
  };

  const onGroup = () => {
    findGroups();
  };

  // Highlight all points with the color of the group they belong to,
  // And draw a larger hollow circle at the centroid of each group.
  const drawGroups = (
    points: Point[],
    assignments: number[],
    centerPoints: Point[],
    group1IsDefault: boolean,
    group2IsDefault: boolean,
    group1Count: number,
    group2Count: number
  ) => {
    context.clearRect(0, 0, canvas.width, canvas.height);
    drawGrid();

    // Draw points with group assignment color
    for (let idx = 0; idx < points.length; idx++) {
      let color: Color = Color.red;
      if (assignments[idx] === 1) {
        color = Color.blue;
      }

      // If points don't belong to a group, draw in black
      if (
        (assignments[idx] === 0 && !group1IsDefault) ||
        (assignments[idx] === 1 && !group2IsDefault)
      ) {
        drawGroupPoint(points[idx].x, height() - points[idx].y, color);
      } else {
        drawSelectionPoint(points[idx].x, height() - points[idx].y);
      }
    }

    // Draw centroid if group is defined
    if (!group1IsDefault) {
      drawCentroid(centerPoints[0].x, height() - centerPoints[0].y, Color.red, group1Count);
    }

    if (!group2IsDefault) {
      drawCentroid(centerPoints[1].x, height() - centerPoints[1].y, Color.blue, group2Count);
    }
  };
  onMount(() => {
    const engine = enableBackEnd();
    setEngine(engine);
    init();
  });

  return (
    <div>
      <header class={styles.header}>
        <h1 onClick={[toggleLog, null]}>Point Box</h1>
        <h2>Hours of Fun*</h2>
        <p>
          Click points on the grey canvas. Click 'Group' to watch group assignment. Watch how the
          center of the group (dashed larger circle) changes and influences which dots belong to a
          group. Watch for existing points that might change groups.
        </p>
      </header>
      <header class={styles.header}>
        <canvas class={styles.pointCanvas} onMouseDown={mouseDownHandler} id="main-canvas"></canvas>
        <div></div>
      </header>
      <header class={styles.row}>
        <button class={styles.actionButton} onClick={[init, null]}>
          Clear
        </button>
        <button
          class={styles.actionButtonWide}
          disabled={isAutoGroupChecked() || maxReached()}
          onClick={[onGroup, null]}
        >
          {prompt()}
        </button>
        <label class={styles.label}>Auto</label>
        <input
          class={styles.largeCheckbox}
          type="checkbox"
          disabled={maxReached()}
          checked={isAutoGroupChecked()}
          onChange={autoGroupChangedHandler}
        ></input>
      </header>
      <header class={styles.row}>
        <span class={styles.label}>Current point</span>
        <select size="1" class={styles.wideSelectList}>
          <Show when={!pointStorage.points.length}>
            <option class={styles.large} selected>
              --Points--
            </option>
          </Show>
          <For each={pointStorage.points}>
            {(item, index) => (
              <option class={styles.large} selected value={index()}>
                {item.toString()}
              </option>
            )}
          </For>
        </select>
      </header>
      <div class={styles.header}>
        <div class={styles.monoLabelRed}>{centroid1()}</div>
        <div class={styles.monoLabelBlue}>{centroid2()}</div>
      </div>
      <hr />

      <span onClick={[functionDemo, null]}>
        *Some have argued 'Minutes of Fun', but agree to disagree.
      </span>
      <span onClick={[logMemory, null]} class={styles.right}>
        {engine()} backend
      </span>
      <div>
        <AboutBox></AboutBox>
      </div>
    </div>
  );
};

export default App;
