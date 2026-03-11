interface Props {
  questions: string[];
  onSelect: (question: string) => void;
}

export default function CommonQuestions({ questions, onSelect }: Props): React.JSX.Element {
  return (
    <div className="common-questions">
      <p className="common-questions-label">Common Questions:</p>
      <div className="common-questions-grid">
        {questions.map((q) => (
          <button
            key={q}
            className="common-question-pill"
            onClick={() => onSelect(q)}
            type="button"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
