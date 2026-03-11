import astronautSvg from "@/images/astronaut1.svg";
import styles from "./PlayerPlaceholder.module.css";

export default function PlayerPlaceholder(): React.JSX.Element {
  return (
    <div className={styles.playerPlaceholder}>
      <div className={styles.placeholderContent}>
        <div className={styles.astronautSvgContainer}>
          <img src={astronautSvg} alt="Astronaut illustration" className={styles.astronautSvg} />
        </div>
        <p>Select a question to watch the response</p>
      </div>
    </div>
  );
}
