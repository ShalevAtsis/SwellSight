import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ocean: {
          950: "#041018",
          900: "#0a1f2e",
          800: "#123347",
          700: "#1a4660",
        },
        swell: {
          500: "#2dd4bf",
          400: "#5eead4",
          300: "#99f6e4",
        },
        foam: {
          50: "#f0fdfa",
          100: "#ccfbf1",
          200: "#a7f3d0",
          300: "#6ee7b7",
          400: "#34d399",
          500: "#94a3b8",
          600: "#64748b",
        },
      },
      fontFamily: {
        display: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
