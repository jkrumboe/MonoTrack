import { defineConfig } from "vitest/config";

// The eval tests run the full 3D solver over every scenario; give them time.
export default defineConfig({
  test: {
    include: ["src/**/*.test.ts"],
    testTimeout: 180_000,
  },
});
