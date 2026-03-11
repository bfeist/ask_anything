import type { ConnectorGeometry } from "@/utils/connector";

interface Props {
  connector: ConnectorGeometry;
}

export default function ConnectorSvg({ connector }: Props): React.JSX.Element {
  return (
    <svg
      className="selection-connector"
      viewBox={`0 0 ${connector.w} ${connector.h}`}
      aria-hidden="true"
    >
      <path d={connector.d} />
    </svg>
  );
}
