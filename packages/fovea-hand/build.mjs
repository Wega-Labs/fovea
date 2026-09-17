import { copyFileSync, cpSync, mkdirSync } from 'node:fs'
copyFileSync('src/hand_landmarker.task', 'dist/hand_landmarker.task')
copyFileSync('src/LICENSE.md', 'dist/MODEL_LICENSE.md')
mkdirSync('dist/LICENSES', { recursive: true })
cpSync('licenses', 'dist/LICENSES', { recursive: true })
