module.exports = {
  extends: ["stylelint-config-standard"],
  rules: {
    "at-rule-no-unknown": [
      true,
      {
        ignoreAtRules: ["value", "import", "export"],
      },
    ],
    "font-family-name-quotes": "always-where-recommended",
    "color-hex-length": "long",
    "shorthand-property-no-redundant-values": null,
    "declaration-block-no-redundant-longhand-properties": null,
    "comment-empty-line-before": null,
    "rule-empty-line-before": [
      "always-multi-line",
      {
        except: ["first-nested"],
        ignore: ["after-comment"],
      },
    ],
    "no-duplicate-selectors": true,
    "color-no-invalid-hex": true,
    "font-family-no-duplicate-names": true,
    "function-calc-no-unspaced-operator": true,
    "unit-no-unknown": true,
    "property-no-unknown": true,
    "declaration-block-no-duplicate-properties": true,
    "custom-property-pattern": "^([a-z][a-z0-9]*)(-[a-z0-9]+)*$",
    "selector-class-pattern": [
      "^([a-z][a-zA-Z0-9]+|([a-z][a-z0-9]*)(-[a-z0-9]+)*)$",
      {
        message:
          "Classname should be camelCase (e.g. myClass) for CSS Modules or kebab-case (e.g. my-class) for global CSS.",
      },
    ],
  },
  overrides: [
    {
      files: ["**/*.css", "!**/*.module.css"],
      rules: {
        "selector-class-pattern": [
          "^([a-z][a-z0-9]*)(-[a-z0-9]+)*$",
          {
            message: "Selector should be in kebab-case (e.g. .my-class)",
          },
        ],
      },
    },
    {
      files: ["**/*.module.css"],
      rules: {
        "selector-class-pattern": [
          "^([a-z][a-zA-Z0-9]+)$",
          {
            resolveNestedSelectors: true,
            message: "Class selectors in CSS Modules must be camelCase (e.g. .myClass)",
          },
        ],
        "no-descending-specificity": null,
      },
    },
  ],
};
