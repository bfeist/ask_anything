import astronautSvg from "@/images/astronaut1.svg";

export default function PlayerPlaceholder(): React.JSX.Element {
  return (
    <div className="player-placeholder">
      <div className="placeholder-content">
        <div className="astronaut-svg-container">
          <img src={astronautSvg} alt="Astronaut illustration" className="astronaut-svg" />
        </div>
        <p>Select a question to watch the response</p>
      </div>
    </div>
  );
}
