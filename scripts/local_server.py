"""PC camera testing: always mock the robot, even without --dry-run."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'server' / 'src'), str(root / 'shared' / 'src')]

if __name__ == '__main__':
    from gesture_server.main import run
    if not any(arg == '--host' or arg.startswith('--host=') for arg in sys.argv):
        sys.argv.extend(['--host', '127.0.0.1'])
    sys.argv.append('--dry-run')
    run()
