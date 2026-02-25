/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_PROD_DATA: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
