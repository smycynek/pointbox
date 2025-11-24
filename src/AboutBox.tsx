import { createSignal, Show } from 'solid-js';

import { Portal } from 'solid-js/web';
import styles from './App.module.css';
export const AboutBox = () => {
  const [isOpen, setIsOpen] = createSignal(false);
  const closeDialog = () => setIsOpen(false);
  return (
    <>
      <button class={styles.aboutLink} onClick={() => setIsOpen(true)}>
        About...
      </button>
      <Show when={isOpen()}>
        <Portal>
          <div class={styles.aboutDialogBackground} onClick={closeDialog}>
            <div
              class={styles.aboutDialog}
              onClick={(e) => e.stopPropagation()} // Don't close clicking inside dialog
            >
              <button class={styles.closeDialogButton} onClick={closeDialog}>
                X
              </button>
              <h1>About Point Box</h1>
              <p>
                This is a simple machine learning application. I created it to show an easy way to
                get started in machine learning with TensorFlowJS and also show some basic theory
                rather than a pre-fabricated fancy demo package that hides all the details. This app
                uses the k-means algorithm to:
              </p>
              <ol class={styles.simpleList}>
                <li>Start with two arbitrary center-point locations.</li>
                <li>Examine points as they are clicked, plotted, and stored.</li>
                <li>Determine how far away all those points are from those two center-points.</li>
                <li>
                  Partition those points into two groups based on which center they are nearer to.
                </li>
                <li>
                  Calculate the new centroid (center point) of each group. (Shown as hollow-dashed
                  circle.)
                </li>
                <li>Make that the new starting center-point for each group.</li>
                <li>Run steps 3-6 again.</li>
              </ol>
              <p>
                You might argue "This isn't machine learning or AI. This is just simple math."
                That's the point :) Taking data, seeing how well it fits to an initial guess, and
                updating your model or to prepare for future data where you'll repeat the process --
                that's part of what learning is.
              </p>
              <p>
                <a href="https://github.com/smycynek/pointbox">
                  https://github.com/smycynek/pointbox
                </a>
              </p>
            </div>
          </div>
        </Portal>
      </Show>
    </>
  );
};
