import { defineConfig } from 'vite';
export default defineConfig({ base: '/Posterior/', worker: { format: 'es' }, build: { target: 'es2022' } });
