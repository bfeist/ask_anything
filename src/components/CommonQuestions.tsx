import styles from "./CommonQuestions.module.css";

interface Props {
  questions: string[];
  onSelect: (question: string) => void;
}

export default function CommonQuestions({ questions, onSelect }: Props): React.JSX.Element {
  return (
    <div className={styles.commonQuestions}>
      <p className={styles.commonQuestionsLabel}>Common Questions:</p>
      <div className={styles.commonQuestionsGrid}>
        {questions.map((q) => (
          <button
            key={q}
            className={styles.commonQuestionPill}
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
