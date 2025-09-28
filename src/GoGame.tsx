import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { produce } from 'immer';
import { useTimer } from 'react-timer-hook';

// ---- Types & Constants ----
enum Stone {
  EMPTY = 0,
  BLACK = 1,
  WHITE = 2
}

type GameMode = 'ai' | 'local' | 'online';
type Difficulty = 'easy' | 'medium' | 'hard';
type GameStatus = 'playing' | 'finished' | 'paused';
type PlayerColor = 'black' | 'white';

interface Position {
  x: number;
  y: number;
}

interface GameMove {
  player: Stone;
  position: Position;
  timestamp: number;
  captures: number;
  isPass?: boolean;
  boardState: Stone[][];
}

interface Captures {
  black: number;
  white: number;
}

interface GameScore {
  blackScore: number;
  whiteScore: number;
  blackTerritory: number;
  whiteTerritory: number;
  komi: number;
  winner: 'black' | 'white' | 'draw';
}

interface GameSettings {
  boardSize: number;
  komi: number;
  handicap: number;
  difficulty: Difficulty;
  timePerMove: number;
  humanColor: PlayerColor;
}

interface SGFGame {
  moves: string;
  boardSize: number;
  komi: number;
  result?: string;
  playerBlack?: string;
  playerWhite?: string;
  date?: string;
}

// MCTS Node for AI
interface MCTSNode {
  position: Position | null;
  wins: number;
  visits: number;
  children: MCTSNode[];
  parent: MCTSNode | null;
  untriedMoves: Position[];
  playerToMove: Stone;
}

// Constants
const BOARD_SIZES = [9, 13, 19] as const;
const DEFAULT_SETTINGS: GameSettings = {
  boardSize: 9,
  komi: 6.5,
  handicap: 0,
  difficulty: 'medium',
  timePerMove: 30,
  humanColor: 'black'
};

// Generate star points dynamically
function generateStarPoints(boardSize: number): Position[] {
  const points: Position[] = [];
  
  if (boardSize === 9) {
    points.push({ x: 2, y: 2 }, { x: 2, y: 6 }, { x: 6, y: 2 }, { x: 6, y: 6 }, { x: 4, y: 4 });
  } else if (boardSize === 13) {
    points.push({ x: 3, y: 3 }, { x: 3, y: 9 }, { x: 9, y: 3 }, { x: 9, y: 9 }, { x: 6, y: 6 });
  } else if (boardSize === 19) {
    const edge = 3;
    const center = 9;
    const far = 15;
    points.push(
      { x: edge, y: edge }, { x: edge, y: center }, { x: edge, y: far },
      { x: center, y: edge }, { x: center, y: center }, { x: center, y: far },
      { x: far, y: edge }, { x: far, y: center }, { x: far, y: far }
    );
  }
  
  return points;
}

// Tutorial messages
const TUTORIAL_MESSAGES = {
  placement: "Click vào bàn cờ để đặt quân. Quân của bạn cần có ít nhất 1 'khí' (ô trống kề bên).",
  capture: "Bắt quân đối phương bằng cách vây hết tất cả các 'khí' của nhóm quân đó.",
  ko: "Luật Ko: Không được đặt quân tạo ra tình huống lặp lại ngay lập tức.",
  pass: "Pass khi không có nước đi có lợi. Hai lần pass liên tiếp kết thúc ván cờ.",
  territory: "Vùng đất là các ô trống được bao quanh hoàn toàn bởi quân cùng màu.",
  scoring: "Điểm = Vùng đất + Quân bắt được + Komi (cho quân trắng)"
};

// ---- Utility Functions ----
function makeEmptyBoard(size: number): Stone[][] {
  return Array.from({ length: size }, () => Array<Stone>(size).fill(Stone.EMPTY));
}

function inBounds(x: number, y: number, size: number): boolean {
  return x >= 0 && y >= 0 && x < size && y < size;
}

function getNeighbors(x: number, y: number, size: number): Position[] {
  const directions = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  return directions
    .map(([dx, dy]) => ({ x: x + dx, y: y + dy }))
    .filter(pos => inBounds(pos.x, pos.y, size));
}

function getGroupAndLiberties(board: Stone[][], x: number, y: number): { 
  group: Position[]; 
  liberties: Set<string> 
} {
  const size = board.length;
  const color = board[y][x];
  if (color === Stone.EMPTY) return { group: [], liberties: new Set() };

  const visited = new Set<string>();
  const liberties = new Set<string>();
  const group: Position[] = [];
  const key = (px: number, py: number) => `${px},${py}`;

  const stack: Position[] = [{ x, y }];
  visited.add(key(x, y));

  while (stack.length > 0) {
    const current = stack.pop()!;
    group.push(current);

    for (const neighbor of getNeighbors(current.x, current.y, size)) {
      const neighborColor = board[neighbor.y][neighbor.x];
      if (neighborColor === Stone.EMPTY) {
        liberties.add(key(neighbor.x, neighbor.y));
      } else if (neighborColor === color && !visited.has(key(neighbor.x, neighbor.y))) {
        visited.add(key(neighbor.x, neighbor.y));
        stack.push(neighbor);
      }
    }
  }

  return { group, liberties };
}

function tryPlay(
  board: Stone[][], 
  x: number, 
  y: number, 
  color: Stone,
  koBoard: Stone[][] | null
): { 
  legal: boolean; 
  board?: Stone[][]; 
  captures?: number;
  capturedPositions?: Position[];
} {
  const size = board.length;

  if (board[y][x] !== Stone.EMPTY) return { legal: false };

  const testBoard = produce(board, draft => {
    draft[y][x] = color;
  });

  const opponent: Stone = color === Stone.BLACK ? Stone.WHITE : Stone.BLACK;
  let totalCaptures = 0;
  const capturedPositions: Position[] = [];

  const finalBoard = produce(testBoard, draft => {
    for (const neighbor of getNeighbors(x, y, size)) {
      if (draft[neighbor.y][neighbor.x] === opponent) {
        const { group, liberties } = getGroupAndLiberties(draft, neighbor.x, neighbor.y);
        if (liberties.size === 0) {
          totalCaptures += group.length;
          group.forEach(pos => {
            draft[pos.y][pos.x] = Stone.EMPTY;
            capturedPositions.push(pos);
          });
        }
      }
    }
  });

  const { liberties } = getGroupAndLiberties(finalBoard, x, y);
  if (liberties.size === 0 && totalCaptures === 0) {
    return { legal: false };
  }

  if (koBoard && totalCaptures === 1) {
    let isSameAsKo = true;
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        if (finalBoard[i][j] !== koBoard[i][j]) {
          isSameAsKo = false;
          break;
        }
      }
      if (!isSameAsKo) break;
    }
    if (isSameAsKo) {
      return { legal: false };
    }
  }

  return { 
    legal: true, 
    board: finalBoard, 
    captures: totalCaptures,
    capturedPositions 
  };
}

// ---- MCTS AI Implementation ----
class MCTSEngine {
  private readonly maxSimulations: number;
  private readonly explorationConstant: number = Math.sqrt(2);

  constructor(difficulty: Difficulty) {
    switch (difficulty) {
      case 'easy':
        this.maxSimulations = 100;
        break;
      case 'medium':
        this.maxSimulations = 500;
        break;
      case 'hard':
        this.maxSimulations = 1000;
        break;
    }
  }

  selectMove(board: Stone[][], color: Stone, koBoard: Stone[][] | null): Position | null {
    const root = this.createNode(null, null, color, board);
    
    for (let i = 0; i < this.maxSimulations; i++) {
      const node = this.select(root, board, koBoard);
      const result = this.simulate(node, board, color, koBoard);
      this.backpropagate(node, result);
    }

    // Select best move
    let bestNode: MCTSNode | null = null;
    let bestVisits = -1;

    for (const child of root.children) {
      if (child.visits > bestVisits) {
        bestVisits = child.visits;
        bestNode = child;
      }
    }

    return bestNode?.position || null;
  }

  private createNode(
    parent: MCTSNode | null,
    position: Position | null,
    playerToMove: Stone,
    board: Stone[][]
  ): MCTSNode {
    const untriedMoves = this.getPossibleMoves(board, playerToMove);
    return {
      position,
      wins: 0,
      visits: 0,
      children: [],
      parent,
      untriedMoves,
      playerToMove
    };
  }

  private getPossibleMoves(board: Stone[][], color: Stone): Position[] {
    const moves: Position[] = [];
    const size = board.length;

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (board[y][x] === Stone.EMPTY) {
          const result = tryPlay(board, x, y, color, null);
          if (result.legal) {
            moves.push({ x, y });
          }
        }
      }
    }

    return moves;
  }

  private select(node: MCTSNode, board: Stone[][], koBoard: Stone[][] | null): MCTSNode {
    let current = node;
    let currentBoard = [...board.map(row => [...row])];

    while (current.untriedMoves.length === 0 && current.children.length > 0) {
      current = this.selectBestChild(current);
      if (current.position) {
        const result = tryPlay(currentBoard, current.position.x, current.position.y, 
          current.parent!.playerToMove, koBoard);
        if (result.board) {
          currentBoard = result.board;
        }
      }
    }

    if (current.untriedMoves.length > 0) {
      const moveIndex = Math.floor(Math.random() * current.untriedMoves.length);
      const move = current.untriedMoves[moveIndex];
      current.untriedMoves.splice(moveIndex, 1);

      const nextPlayer = current.playerToMove === Stone.BLACK ? Stone.WHITE : Stone.BLACK;
      const child = this.createNode(current, move, nextPlayer, currentBoard);
      current.children.push(child);
      return child;
    }

    return current;
  }

  private selectBestChild(node: MCTSNode): MCTSNode {
    let bestChild: MCTSNode | null = null;
    let bestValue = -Infinity;

    for (const child of node.children) {
      const exploitation = child.wins / (child.visits + 1e-10);
      const exploration = Math.sqrt(Math.log(node.visits + 1) / (child.visits + 1e-10));
      const value = exploitation + this.explorationConstant * exploration;

      if (value > bestValue) {
        bestValue = value;
        bestChild = child;
      }
    }

    return bestChild!;
  }

  private simulate(node: MCTSNode, board: Stone[][], originalColor: Stone, koBoard: Stone[][] | null): number {
    let simulationBoard = [...board.map(row => [...row])];
    let currentPlayer = node.playerToMove;
    let passes = 0;

    for (let i = 0; i < 100; i++) {
      const moves = this.getPossibleMoves(simulationBoard, currentPlayer);
      
      if (moves.length === 0) {
        passes++;
        if (passes >= 2) break;
      } else {
        passes = 0;
        const move = moves[Math.floor(Math.random() * moves.length)];
        const result = tryPlay(simulationBoard, move.x, move.y, currentPlayer, koBoard);
        if (result.board) {
          simulationBoard = result.board;
        }
      }

      currentPlayer = currentPlayer === Stone.BLACK ? Stone.WHITE : Stone.BLACK;
    }

    // Simple evaluation
    const score = this.evaluateBoard(simulationBoard, originalColor);
    return score > 0 ? 1 : 0;
  }

  private evaluateBoard(board: Stone[][], color: Stone): number {
    let myStones = 0;
    let oppStones = 0;
    const opponent = color === Stone.BLACK ? Stone.WHITE : Stone.BLACK;

    for (const row of board) {
      for (const stone of row) {
        if (stone === color) myStones++;
        else if (stone === opponent) oppStones++;
      }
    }

    return myStones - oppStones;
  }

  private backpropagate(node: MCTSNode | null, result: number): void {
    while (node !== null) {
      node.visits++;
      node.wins += result;
      result = 1 - result; // Flip for opponent
      node = node.parent;
    }
  }
}

// ---- Optimized AI with fallback ----
function getRelevantPositions(board: Stone[][], maxDistance: number = 2): Position[] {
  const size = board.length;
  const relevantPositions = new Set<string>();

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (board[y][x] !== Stone.EMPTY) {
        for (let dy = -maxDistance; dy <= maxDistance; dy++) {
          for (let dx = -maxDistance; dx <= maxDistance; dx++) {
            const nx = x + dx;
            const ny = y + dy;
            if (inBounds(nx, ny, size) && board[ny][nx] === Stone.EMPTY) {
              relevantPositions.add(`${nx},${ny}`);
            }
          }
        }
      }
    }
  }

  if (relevantPositions.size < 10) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (board[y][x] === Stone.EMPTY) {
          relevantPositions.add(`${x},${y}`);
        }
      }
    }
  }

  return Array.from(relevantPositions).map(key => {
    const [x, y] = key.split(',').map(Number);
    return { x, y };
  });
}

function calculateMoveScore(
  board: Stone[][], 
  x: number, 
  y: number, 
  color: Stone, 
  difficulty: Difficulty,
  koBoard: Stone[][] | null
): number {
  const size = board.length;
  let score = 0;

  const result = tryPlay(board, x, y, color, koBoard);
  if (!result.legal) return -Infinity;

  const captures = result.captures || 0;
  const centerX = Math.floor(size / 2);
  const centerY = Math.floor(size / 2);
  const distanceFromCenter = Math.abs(centerX - x) + Math.abs(centerY - y);

  score += captures * 20;
  score -= distanceFromCenter * 3;

  if (difficulty === 'easy') {
    score += Math.random() * 30 - 15;
  } else {
    const neighbors = getNeighbors(x, y, size);
    const friendlyNeighbors = neighbors.filter(n => board[n.y][n.x] === color).length;
    const opponentNeighbors = neighbors.filter(n => 
      board[n.y][n.x] === (color === Stone.BLACK ? Stone.WHITE : Stone.BLACK)).length;

    score += friendlyNeighbors * 8;
    score += opponentNeighbors * 6;

    if (difficulty === 'hard') {
      const cornerBonus = (x === 0 || x === size-1) && (y === 0 || y === size-1) ? 5 : 0;
      const sideBonus = (x === 0 || x === size-1 || y === 0 || y === size-1) ? 3 : 0;
      const eyePotential = neighbors.filter(n => board[n.y][n.x] === color).length >= 3 ? 10 : 0;
      
      score += cornerBonus + sideBonus + eyePotential;
    }
  }

  return score;
}

function pickAiMove(
  board: Stone[][], 
  color: Stone, 
  difficulty: Difficulty,
  koBoard: Stone[][] | null
): Position | null {
  // Use MCTS for hard difficulty
  if (difficulty === 'hard') {
    const mcts = new MCTSEngine(difficulty);
    return mcts.selectMove(board, color, koBoard);
  }

  // Use heuristic for easy/medium
  const relevantPositions = getRelevantPositions(board, difficulty === 'medium' ? 3 : 2);
  const candidates: { position: Position; score: number }[] = [];

  for (const pos of relevantPositions) {
    const score = calculateMoveScore(board, pos.x, pos.y, color, difficulty, koBoard);
    if (score > -Infinity) {
      candidates.push({ position: pos, score });
    }
  }

  if (candidates.length === 0) return null;

  candidates.sort((a, b) => b.score - a.score);

  let selectedIndex = 0;
  if (difficulty === 'easy' && candidates.length > 3) {
    selectedIndex = Math.floor(Math.random() * Math.min(5, candidates.length));
  } else if (difficulty === 'medium' && candidates.length > 2) {
    selectedIndex = Math.floor(Math.random() * Math.min(3, candidates.length));
  }

  return candidates[selectedIndex].position;
}

// ---- Enhanced SGF Functions ----
function positionToSGF(pos: Position, boardSize: number): string {
  if (pos.x === -1 && pos.y === -1) return '';
  const x = String.fromCharCode(97 + pos.x);
  const y = String.fromCharCode(97 + pos.y);
  return x + y;
}

function sgfToPosition(sgf: string): Position {
  if (sgf === '') return { x: -1, y: -1 };
  const x = sgf.charCodeAt(0) - 97;
  const y = sgf.charCodeAt(1) - 97;
  return { x, y };
}

function exportToSGF(
  moves: GameMove[],
  settings: GameSettings,
  score?: GameScore
): string {
  let sgf = '(;FF[4]GM[1]SZ[' + settings.boardSize + ']';
  sgf += 'KM[' + settings.komi + ']';

  if (score) {
    const result = score.winner === 'black' 
      ? `B+${(score.blackScore - score.whiteScore).toFixed(1)}`
      : `W+${(score.whiteScore - score.blackScore).toFixed(1)}`;
    sgf += 'RE[' + result + ']';
  }

  for (const move of moves) {
    const color = move.player === Stone.BLACK ? 'B' : 'W';
    const coord = positionToSGF(move.position, settings.boardSize);
    sgf += ';' + color + '[' + coord + ']';
  }

  sgf += ')';
  return sgf;
}

function importFromSGF(sgfContent: string): { 
  moves: { position: Position; player: Stone }[]; 
  boardSize: number; 
  komi: number 
} | null {
  try {
    // Remove whitespace and normalize
    const content = sgfContent.replace(/\s+/g, ' ').trim();
    
    // Parse board size
    const sizeMatch = content.match(/SZ```math
(\d+)```/);
    const boardSize = sizeMatch ? parseInt(sizeMatch[1]) : 19;

    // Parse komi
    const komiMatch = content.match(/KM```math
([0-9.-]+)```/);
    const komi = komiMatch ? parseFloat(komiMatch[1]) : 6.5;

    // Parse moves
    const moves: { position: Position; player: Stone }[] = [];
    const movePattern = /;([BW])```math
([a-s]{0,2})```/g;
    let match;

    while ((match = movePattern.exec(content)) !== null) {
      const player = match[1] === 'B' ? Stone.BLACK : Stone.WHITE;
      const position = sgfToPosition(match[2]);
      moves.push({ position, player });
    }

    return { moves, boardSize, komi };
  } catch (error) {
    console.error('SGF parse error:', error);
    return null;
  }
}

// ---- UI Components ----
const StoneComponent = React.memo(({ 
  color, 
  size = 'medium', 
  isLastMove = false,
  isAnimating = false 
}: { 
  color: Stone;
  size?: 'small' | 'medium' | 'large';
  isLastMove?: boolean;
  isAnimating?: boolean;
}) => {
  const sizeClasses = {
    small: 'w-3 h-3 xs:w-2.5 xs:h-2.5 sm:w-4 sm:h-4',
    medium: 'w-5 h-5 xs:w-4 xs:h-4 sm:w-7 sm:h-7',
    large: 'w-8 h-8 xs:w-6 xs:h-6 sm:w-10 sm:h-10'
  };

  if (color === Stone.EMPTY) return null;

  return (
    <div 
      className={`
        rounded-full transition-all duration-300 relative 
        ${sizeClasses[size]} 
        ${color === Stone.BLACK 
          ? 'bg-gradient-to-br from-gray-700 via-gray-800 to-black shadow-stone-black' 
          : 'bg-gradient-to-br from-gray-100 via-white to-gray-200 shadow-stone-white'} 
        ${isLastMove ? 'ring-2 ring-red-500 ring-opacity-60' : ''} 
        ${isAnimating ? 'animate-pulse scale-110' : ''}
      `}
      style={{
        boxShadow: color === Stone.BLACK 
          ? '2px 2px 4px rgba(0,0,0,0.5), inset -1px -1px 2px rgba(255,255,255,0.1)' 
          : '2px 2px 4px rgba(0,0,0,0.3), inset -1px -1px 2px rgba(0,0,0,0.1)'
      }}
    >
      <div className={`absolute top-0.5 left-0.5 xs:top-0.5 xs:left-0.5 sm:top-1 sm:left-1 w-1.5 h-1.5 xs:w-1 xs:h-1 sm:w-2 sm:h-2 rounded-full ${
        color === Stone.BLACK ? 'bg-gray-600 opacity-30' : 'bg-white opacity-60'
      }`} />
    </div>
  );
});

const BoardCell = React.memo(({
  position,
  stone,
  boardSize,
  cellSize,
  isHovered,
  isValidMove,
  isStarPoint,
  isLastMove,
  onHover,
  onClick,
  currentPlayer
}: {
  position: Position;
  stone: Stone;
  boardSize: number;
  cellSize: number;
  isHovered: boolean;
  isValidMove: boolean;
  isStarPoint: boolean;
  isLastMove: boolean;
  onHover: (position: Position | null) => void;
  onClick: (position: Position) => void;
  currentPlayer: Stone;
}) => {
  const { x, y } = position;

  return (
    <div
      className="absolute flex items-center justify-center cursor-pointer"
      style={{
        left: x * cellSize - cellSize/2,
        top: y * cellSize - cellSize/2,
        width: cellSize,
        height: cellSize,
      }}
      onMouseEnter={() => onHover(position)}
      onMouseLeave={() => onHover(null)}
      onClick={() => {
        if (isValidMove && stone === Stone.EMPTY) {
          onClick(position);
        }
      }}
    >
      {isStarPoint && stone === Stone.EMPTY && (
        <div className="absolute w-1.5 h-1.5 xs:w-1 xs:h-1 sm:w-2 sm:h-2 rounded-full bg-amber-900 z-10" />
      )}

      {isHovered && stone === Stone.EMPTY && isValidMove && (
        <div className={`absolute w-4 h-4 xs:w-3 xs:h-3 sm:w-6 sm:h-6 rounded-full opacity-40 z-20 ${
          currentPlayer === Stone.BLACK ? 'bg-gray-900' : 'bg-white border border-gray-400'
        }`} />
      )}

      {stone !== Stone.EMPTY && (
        <div className="absolute z-30">
          <StoneComponent color={stone} isLastMove={isLastMove} />
        </div>
      )}
    </div>
  );
});

const TimerDisplay = ({ 
  expiryTimestamp, 
  onExpire,
  isActive,
  color
}: {
  expiryTimestamp: Date;
  onExpire: () => void;
  isActive: boolean;
  color: 'black' | 'white';
}) => {
  const {
    seconds,
    minutes,
    pause,
    resume,
    restart
  } = useTimer({ 
    expiryTimestamp, 
    onExpire,
    autoStart: isActive 
  });

  useEffect(() => {
    if (isActive) {
      resume();
    } else {
      pause();
    }
  }, [isActive, pause, resume]);

  return (
    <div className={`px-2 py-1 rounded ${isActive ? 'bg-red-600 text-white' : 'bg-gray-200'}`}>
      {color === 'black' ? '⚫' : '⚪'} {minutes}:{seconds.toString().padStart(2, '0')}
    </div>
  );
};

const Tooltip = ({ 
  children, 
  content 
}: { 
  children: React.ReactNode; 
  content: string 
}) => {
  const [show, setShow] = useState(false);

  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
      >
        {children}
      </div>
      {show && (
        <div className="absolute z-50 bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 text-sm text-white bg-gray-800 rounded-lg whitespace-nowrap">
          {content}
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1">
            <div className="border-4 border-transparent border-t-gray-800" />
          </div>
        </div>
      )}
    </div>
  );
};

// ---- Main Game Component ----
const GoGame = () => {
  const [gameMode, setGameMode] = useState<GameMode>('local');
  const [settings, setSettings] = useState<GameSettings>(DEFAULT_SETTINGS);
  const [board, setBoard] = useState<Stone[][]>(() => makeEmptyBoard(settings.boardSize));
  const [currentPlayer, setCurrentPlayer] = useState<Stone>(Stone.BLACK);
  const [captures, setCaptures] = useState<Captures>({ black: 0, white: 0 });
  const [moveHistory, setMoveHistory] = useState<GameMove[]>([]);
  const [gameStatus, setGameStatus] = useState<GameStatus>('playing');
  const [passCount, setPassCount] = useState(0);
  const [hoverPosition, setHoverPosition] = useState<Position | null>(null);
  const [lastMove, setLastMove] = useState<Position | null>(null);
  const [showScore, setShowScore] = useState(false);
  const [gameScore, setGameScore] = useState<GameScore | null>(null);
  const [animatingCaptures, setAnimatingCaptures] = useState<Position[]>([]);
  const [showTutorial, setShowTutorial] = useState(false);
  const [isLoadingAI, setIsLoadingAI] = useState(false);

  const koHistoryRef = useRef<Stone[][] | null>(null);
  const aiMoveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const getAIColor = useCallback((): Stone => {
    return settings.humanColor === 'black' ? Stone.WHITE : Stone.BLACK;
  }, [settings.humanColor]);

  const initializeGame = useCallback(() => {
    const newBoard = makeEmptyBoard(settings.boardSize);
    setBoard(newBoard);
    setCurrentPlayer(Stone.BLACK);
    setCaptures({ black: 0, white: 0 });
    setMoveHistory([]);
    setGameStatus('playing');
    setPassCount(0);
    setLastMove(null);
    setShowScore(false);
    setGameScore(null);
    setAnimatingCaptures([]);
    koHistoryRef.current = null;

    if (gameMode === 'ai' && settings.humanColor === 'white') {
      setTimeout(() => {
        const aiMove = pickAiMove(newBoard, Stone.BLACK, settings.difficulty, null);
        if (aiMove) {
          handlePlaceStone(aiMove);
        }
      }, 500);
    }
  }, [settings.boardSize, settings.humanColor, settings.difficulty, gameMode]);

  const cleanup = useCallback(() => {
    if (aiMoveTimeoutRef.current) {
      clearTimeout(aiMoveTimeoutRef.current);
      aiMoveTimeoutRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, []);

  const isValidMove = useCallback((position: Position): boolean => {
    const { x, y } = position;
    
    if (!inBounds(x, y, board.length)) return false;
    if (board[y][x] !== Stone.EMPTY) return false;
    if (gameStatus !== 'playing') return false;

    if (gameMode === 'ai') {
      const aiColor = getAIColor();
      if (currentPlayer === aiColor) return false;
    }

    const result = tryPlay(board, x, y, currentPlayer, koHistoryRef.current);
    return result.legal;
  }, [board, gameStatus, currentPlayer, gameMode, getAIColor]);

  const calculateTerritory = useCallback((): { black: number; white: number; seki: Position[] } => {
    const size = board.length;
    const visited = new Set<string>();
    let blackTerritory = 0;
    let whiteTerritory = 0;
    const sekiPositions: Position[] = [];

    const floodFill = (startX: number, startY: number): { 
      points: Position[]; 
      owner: Stone;
      isSeki: boolean;
    } => {
      if (visited.has(`${startX},${startY}`) || board[startY][startX] !== Stone.EMPTY) {
        return { points: [], owner: Stone.EMPTY, isSeki: false };
      }

      const queue: Position[] = [{ x: startX, y: startY }];
      const territory: Position[] = [];
      const borderStones = new Set<Stone>();
      const borderGroups = new Set<string>();

      while (queue.length > 0) {
        const current = queue.shift()!;
        if (visited.has(`${current.x},${current.y}`)) continue;
        
        visited.add(`${current.x},${current.y}`);
        territory.push(current);

        for (const neighbor of getNeighbors(current.x, current.y, size)) {
          const neighborStone = board[neighbor.y][neighbor.x];
          if (neighborStone === Stone.EMPTY && !visited.has(`${neighbor.x},${neighbor.y}`)) {
            queue.push(neighbor);
          } else if (neighborStone !== Stone.EMPTY) {
            borderStones.add(neighborStone);
            const { group } = getGroupAndLiberties(board, neighbor.x, neighbor.y);
            borderGroups.add(JSON.stringify(group[0]));
          }
        }
      }

      let owner: Stone = Stone.EMPTY;
      let isSeki = false;
      
      if (borderStones.size === 2 && borderGroups.size === 2) {
        isSeki = true;
      } else if (borderStones.size === 1) {
        owner = Array.from(borderStones)[0];
      }

      return { points: territory, owner, isSeki };
    };

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (!visited.has(`${x},${y}`) && board[y][x] === Stone.EMPTY) {
          const territory = floodFill(x, y);
          if (territory.isSeki) {
            sekiPositions.push(...territory.points);
          } else if (territory.owner === Stone.BLACK) {
            blackTerritory += territory.points.length;
          } else if (territory.owner === Stone.WHITE) {
            whiteTerritory += territory.points.length;
          }
        }
      }
    }

    return { black: blackTerritory, white: whiteTerritory, seki: sekiPositions };
  }, [board]);

  const calculateScore = useCallback((): GameScore => {
    const territory = calculateTerritory();
    const stoneCount = board.flat().reduce((acc, stone) => {
      if (stone === Stone.BLACK) acc.black++;
      if (stone === Stone.WHITE) acc.white++;
      return acc;
    }, { black: 0, white: 0 });

    const blackScore = territory.black + stoneCount.black + captures.black;
    const whiteScore = territory.white + stoneCount.white + captures.white + settings.komi;

    let winner: 'black' | 'white' | 'draw';
    if (Math.abs(blackScore - whiteScore) < 0.1) {
      winner = 'draw';
    } else {
      winner = blackScore > whiteScore ? 'black' : 'white';
    }

    return {
      blackScore,
      whiteScore,
      blackTerritory: territory.black,
      whiteTerritory: territory.white,
      komi: settings.komi,
      winner
    };
  }, [board, captures, settings.komi, calculateTerritory]);

  const handleAIMove = useCallback(() => {
    if (gameStatus !== 'playing') return;
    
    const aiColor = getAIColor();
    if (currentPlayer !== aiColor) return;

    setIsLoadingAI(true);

    animationFrameRef.current = requestAnimationFrame(() => {
      const aiMove = pickAiMove(board, aiColor, settings.difficulty, koHistoryRef.current);
      setIsLoadingAI(false);
      
      if (aiMove) {
        handlePlaceStone(aiMove);
      } else {
        handlePass();
      }
    });
  }, [board, currentPlayer, gameStatus, settings.difficulty, getAIColor]);

  const handlePlaceStone = useCallback((position: Position) => {
    if (!isValidMove(position) && !(gameMode === 'ai' && currentPlayer === getAIColor())) return;

    const result = tryPlay(board, position.x, position.y, currentPlayer, koHistoryRef.current);
    if (!result.legal || !result.board) return;

    if (result.captures === 1) {
      koHistoryRef.current = board;
    } else {
      koHistoryRef.current = null;
    }

    setBoard(result.board);
    setLastMove(position);

    if (result.captures && result.captures > 0) {
      setCaptures(prev => ({
        ...prev,
        [currentPlayer === Stone.BLACK ? 'black' : 'white']: 
          prev[currentPlayer === Stone.BLACK ? 'black' : 'white'] + result.captures!
      }));
      
      if (result.capturedPositions) {
        setAnimatingCaptures(result.capturedPositions);
        setTimeout(() => setAnimatingCaptures([]), 500);
      }
    }

    const newMove: GameMove = {
      player: currentPlayer,
      position,
      timestamp: Date.now(),
      captures: result.captures || 0,
      boardState: produce(result.board, draft => {})
    };
    setMoveHistory(prev => [...prev, newMove]);

    const nextPlayer: Stone = currentPlayer === Stone.BLACK ? Stone.WHITE : Stone.BLACK;
    setCurrentPlayer(nextPlayer);
    setPassCount(0);

    if (gameMode === 'ai' && nextPlayer === getAIColor()) {
      if (aiMoveTimeoutRef.current) {
        clearTimeout(aiMoveTimeoutRef.current);
      }
      aiMoveTimeoutRef.current = setTimeout(() => {
        handleAIMove();
      }, 700);
    }
  }, [board, currentPlayer, gameMode, isValidMove, handleAIMove, getAIColor]);

  const handlePass = useCallback(() => {
    const newPassCount = passCount + 1;
    setPassCount(newPassCount);

    const passMove: GameMove = {
      player: currentPlayer,
      position: { x: -1, y: -1 },
      timestamp: Date.now(),
      captures: 0,
      isPass: true,
      boardState: produce(board, draft => {})
    };
    setMoveHistory(prev => [...prev, passMove]);

    if (newPassCount >= 2) {
      setGameStatus('finished');
      const finalScore = calculateScore();
      setGameScore(finalScore);
      setShowScore(true);
    } else {
      const nextPlayer: Stone = currentPlayer === Stone.BLACK ? Stone.WHITE : Stone.BLACK;
      setCurrentPlayer(nextPlayer);

      if (gameMode === 'ai' && nextPlayer === getAIColor()) {
        if (aiMoveTimeoutRef.current) {
          clearTimeout(aiMoveTimeoutRef.current);
        }
        aiMoveTimeoutRef.current = setTimeout(() => {
          handleAIMove();
        }, 700);
      }
    }
  }, [passCount, currentPlayer, board, gameMode, calculateScore, handleAIMove, getAIColor]);

  const handleUndo = useCallback(() => {
    if (moveHistory.length === 0) return;

    let movesToUndo = 1;
    if (gameMode === 'ai' && moveHistory.length > 1) {
      const lastMove = moveHistory[moveHistory.length - 1];
      const aiColor = getAIColor();
      if (lastMove.player === aiColor) {
        movesToUndo = 1;
      } else {
        movesToUndo = 2;
      }
    }

    for (let i = 0; i < movesToUndo && moveHistory.length > 0; i++) {
      const lastMove = moveHistory[moveHistory.length - 1];
      const newHistory = moveHistory.slice(0, -1);
      
      if (lastMove.isPass) {
        setPassCount(prev => Math.max(0, prev - 1));
      } else {
        const previousMove = newHistory[newHistory.length - 1];
        if (previousMove) {
          setBoard(previousMove.boardState);
        } else {
          setBoard(makeEmptyBoard(settings.boardSize));
        }
        
        if (lastMove.captures > 0) {
          setCaptures(prev => ({
            ...prev,
            [lastMove.player === Stone.BLACK ? 'black' : 'white']: 
              Math.max(0, prev[lastMove.player === Stone.BLACK ? 'black' : 'white'] - lastMove.captures)
          }));
        }
      }

      setMoveHistory(newHistory);
      setCurrentPlayer(lastMove.player);
    }

    if (gameStatus === 'finished') {
      setGameStatus('playing');
      setShowScore(false);
    }

    koHistoryRef.current = null;
  }, [moveHistory, gameStatus, settings.boardSize, gameMode, getAIColor]);

  const saveGame = useCallback(() => {
    const sgf = exportToSGF(moveHistory, settings, gameScore || undefined);
    const blob = new Blob([sgf], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `go_game_${new Date().toISOString()}.sgf`;
    a.click();
    URL.revokeObjectURL(url);
  }, [moveHistory, settings, gameScore]);

  // Enhanced loadGame with replay functionality
  const loadGame = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
      const content = e.target?.result as string;
      const gameData = importFromSGF(content);
      
      if (!gameData) {
        alert('Lỗi: Không thể đọc file SGF. Vui lòng kiểm tra định dạng file.');
        return;
      }

      try {
        // Update settings
        setSettings(prev => ({
          ...prev,
          boardSize: gameData.boardSize,
          komi: gameData.komi
        }));
        
        // Initialize new game
        const newBoard = makeEmptyBoard(gameData.boardSize);
        setBoard(newBoard);
        setMoveHistory([]);
        setCurrentPlayer(Stone.BLACK);
        setCaptures({ black: 0, white: 0 });
        setGameStatus('playing');
        setPassCount(0);
        setLastMove(null);
        koHistoryRef.current = null;

        // Replay moves
        let currentBoard = newBoard;
        let currentCaptures = { black: 0, white: 0 };
        let koBoard: Stone[][] | null = null;
        const newMoveHistory: GameMove[] = [];

        for (const move of gameData.moves) {
          if (move.position.x === -1 && move.position.y === -1) {
            // Handle pass
            const passMove: GameMove = {
              player: move.player,
              position: move.position,
              timestamp: Date.now(),
              captures: 0,
              isPass: true,
              boardState: produce(currentBoard, draft => {})
            };
            newMoveHistory.push(passMove);
          } else {
            // Try to play the move
            const result = tryPlay(currentBoard, move.position.x, move.position.y, move.player, koBoard);
            
            if (!result.legal || !result.board) {
              console.warn(`Invalid move at position (${move.position.x}, ${move.position.y})`);
              continue;
            }

            currentBoard = result.board;
            
            if (result.captures && result.captures > 0) {
              if (move.player === Stone.BLACK) {
                currentCaptures.black += result.captures;
              } else {
                currentCaptures.white += result.captures;
              }
            }

            if (result.captures === 1) {
              koBoard = currentBoard;
            } else {
              koBoard = null;
            }

            const gameMove: GameMove = {
              player: move.player,
              position: move.position,
              timestamp: Date.now(),
              captures: result.captures || 0,
              boardState: produce(currentBoard, draft => {})
            };
            newMoveHistory.push(gameMove);
          }
        }

        // Apply final state
        setBoard(currentBoard);
        setCaptures(currentCaptures);
        setMoveHistory(newMoveHistory);
        
        // Determine current player
        const lastMove = newMoveHistory[newMoveHistory.length - 1];
        if (lastMove) {
          setCurrentPlayer(lastMove.player === Stone.BLACK ? Stone.WHITE : Stone.BLACK);
          if (!lastMove.isPass) {
            setLastMove(lastMove.position);
          }
        }

        // Clear file input
        event.target.value = '';
        
      } catch (error) {
        console.error('Error loading game:', error);
        alert('Lỗi khi tải ván cờ. Vui lòng thử lại.');
      }
    };
    
    reader.onerror = () => {
      alert('Lỗi khi đọc file. Vui lòng thử lại.');
    };
    
    reader.readAsText(file);
  }, []);

  useEffect(() => {
    initializeGame();
  }, [initializeGame]);

  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  useEffect(() => {
    if (gameMode === 'ai' && currentPlayer === getAIColor() && gameStatus === 'playing') {
      if (aiMoveTimeoutRef.current) {
        clearTimeout(aiMoveTimeoutRef.current);
      }
      aiMoveTimeoutRef.current = setTimeout(() => {
        handleAIMove();
      }, 700);
    }
  }, [gameMode, currentPlayer, gameStatus, getAIColor, handleAIMove]);

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === 'p' || e.key === 'P') {
        handlePass();
      } else if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        handleUndo();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [handlePass, handleUndo]);

  const renderBoard = useMemo(() => {
    const size = settings.boardSize;
    const windowWidth = typeof window !== 'undefined' ? window.innerWidth : 500;
    
    // Enhanced responsive sizing for <320px
    let maxDisplaySize: number;
    if (windowWidth < 320) {
      maxDisplaySize = windowWidth - 20;
    } else if (windowWidth < 640) {
      maxDisplaySize = Math.min(windowWidth - 40, 400);
    } else {
      maxDisplaySize = Math.min(windowWidth - 40, 500);
    }
    
    const displaySize = size === 9 
      ? Math.min(400, maxDisplaySize) 
      : size === 13 
        ? Math.min(450, maxDisplaySize)
        : Math.min(500, maxDisplaySize);
        
    const cellSize = displaySize / (size - 1);
    const padding = cellSize / 2;
    const starPoints = generateStarPoints(size);

    return (
      <div 
        className="relative rounded-lg shadow-2xl mx-auto"
        style={{ 
          width: displaySize + padding * 2, 
          height: displaySize + padding * 2,
          background: 'linear-gradient(135deg, #d4a574 0%, #e6c088 50%, #d4a574 100%)',
          padding: `${padding}px`
        }}
      >
        <div 
          className="absolute inset-0 opacity-20 rounded-lg"
          style={{
            backgroundImage: `repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(139,69,19,0.1) 2px, rgba(139,69,19,0.1) 4px)`,
          }}
        />
        
        {Array.from({ length: size }).map((_, index) => (
          <React.Fragment key={`grid-${index}`}>
            <div 
              className="absolute"
              style={{
                left: padding,
                top: padding + index * cellSize,
                width: displaySize,
                height: 1,
                background: 'linear-gradient(90deg, transparent 0%, #8B4513 10%, #8B4513 90%, transparent 100%)',
              }}
            />
            <div 
              className="absolute"
              style={{
                top: padding,
                left: padding + index * cellSize,
                height: displaySize,
                width: 1,
                background: 'linear-gradient(180deg, transparent 0%, #8B4513 10%, #8B4513 90%, transparent 100%)',
              }}
            />
          </React.Fragment>
        ))}

        {board.map((row, y) =>
          row.map((stone, x) => {
            const isStarPoint = starPoints.some(point => point.x === x && point.y === y);
            const isLastMovePosition = lastMove?.x === x && lastMove?.y === y;
            
            return (
              <BoardCell
                key={`cell-${x}-${y}`}
                position={{ x, y }}
                stone={stone}
                boardSize={size}
                cellSize={cellSize}
                isHovered={hoverPosition?.x === x && hoverPosition?.y === y}
                isValidMove={isValidMove({ x, y })}
                isStarPoint={isStarPoint}
                isLastMove={isLastMovePosition}
                onHover={setHoverPosition}
                onClick={handlePlaceStone}
                currentPlayer={currentPlayer}
              />
            );
          })
        )}

        {isLoadingAI && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-10 rounded-lg">
            <div className="bg-white px-4 py-2 rounded-lg shadow-lg">
              <span className="text-sm font-medium">AI đang suy nghĩ...</span>
            </div>
          </div>
        )}
      </div>
    );
  }, [board, settings.boardSize, hoverPosition, lastMove, currentPlayer, isValidMove, handlePlaceStone, isLoadingAI]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 to-gray-300 py-2 xs:py-3 sm:py-8 px-1 xs:px-2 sm:px-4">
      <div className="max-w-7xl mx-auto">
        <div className="bg-gradient-to-r from-gray-800 to-gray-900 text-white p-2 xs:p-3 sm:p-6 rounded-t-xl shadow-lg">
          <div className="flex flex-col xs:flex-row sm:flex-row justify-between items-center gap-2 xs:gap-3 sm:gap-4">
            <h1 className="text-lg xs:text-xl sm:text-3xl font-bold">⚫⚪ Cờ Vây Pro</h1>
            <div className="flex items-center space-x-2 xs:space-x-3 sm:space-x-4">
              <span className="flex items-center gap-1 xs:gap-2">
                <div className={`w-4 h-4 xs:w-5 xs:h-5 sm:w-8 sm:h-8 rounded-full ${
                  currentPlayer === Stone.BLACK 
                    ? 'bg-gradient-to-br from-gray-700 to-black shadow-lg' 
                    : 'bg-gradient-to-br from-white to-gray-200 shadow-lg'
                }`} />
                <span className="text-xs xs:text-sm sm:text-base">
                  Lượt: {currentPlayer === Stone.BLACK ? 'Đen' : 'Trắng'}
                </span>
              </span>
              <Tooltip content="Nhấn ? để xem hướng dẫn">
                <button
                  onClick={() => setShowTutorial(true)}
                  className="p-1 xs:p-1.5 sm:p-2 bg-blue-600 rounded-full hover:bg-blue-700 transition-colors text-xs xs:text-sm"
                >
                  ?
                </button>
              </Tooltip>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-b-xl shadow-lg p-2 xs:p-3 sm:p-6">
          <div className="mb-2 xs:mb-4 sm:mb-6 flex flex-col xs:flex-row sm:flex-row justify-center space-y-1 xs:space-y-0 sm:space-y-0 xs:space-x-2 sm:space-x-4">
            <button
              onClick={() => setGameMode('local')}
              className={`px-3 xs:px-4 sm:px-6 py-1.5 xs:py-2 sm:py-3 rounded-lg font-medium transition-all text-xs xs:text-sm sm:text-base ${
                gameMode === 'local' 
                  ? 'bg-blue-500 text-white shadow-lg transform scale-105' 
                  : 'bg-gray-200 hover:bg-gray-300'
              }`}
            >
              👥 Chơi 2 người
            </button>
            <button
              onClick={() => setGameMode('ai')}
              className={`px-3 xs:px-4 sm:px-6 py-1.5 xs:py-2 sm:py-3 rounded-lg font-medium transition-all text-xs xs:text-sm sm:text-base ${
                gameMode === 'ai' 
                  ? 'bg-blue-500 text-white shadow-lg transform scale-105' 
                  : 'bg-gray-200 hover:bg-gray-300'
              }`}
            >
              🤖 Chơi với AI
            </button>
          </div>

          <div className="flex flex-col lg:flex-row gap-2 xs:gap-4 sm:gap-8">
            <div className="lg:w-80 space-y-2 xs:space-y-3 sm:space-y-4 order-2 lg:order-1">
              <div className="bg-gradient-to-br from-gray-50 to-gray-100 p-2 xs:p-3 sm:p-4 rounded-lg border border-gray-200">
                <h3 className="font-bold mb-1 xs:mb-2 sm:mb-3 text-gray-800 text-xs xs:text-sm sm:text-base">
                  📊 Thông tin ván cờ
                </h3>
                <div className="space-y-1 xs:space-y-1.5 sm:space-y-2 text-xs sm:text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Quân bắt được:</span>
                    <span className="font-medium">
                      ⚫ {captures.black} | ⚪ {captures.white}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Komi:</span>
                    <span className="font-medium">{settings.komi}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Số nước:</span>
                    <span className="font-medium">{moveHistory.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Pass liên tiếp:</span>
                    <span className="font-medium">{passCount}/2</span>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-2 xs:p-3 sm:p-4 rounded-lg border border-blue-200">
                <h3 className="font-bold mb-1 xs:mb-2 sm:mb-3 text-gray-800 text-xs xs:text-sm sm:text-base">
                  ⚙️ Cài đặt
                </h3>
                <div className="space-y-2 xs:space-y-2.5 sm:space-y-3">
                  <div>
                    <label className="block text-xs sm:text-sm font-medium mb-0.5 xs:mb-1 text-gray-700">
                      Kích thước bàn cờ:
                    </label>
                    <select
                      value={settings.boardSize}
                      onChange={(e) => setSettings(prev => ({ ...prev, boardSize: Number(e.target.value) }))}
                      className="w-full p-1.5 xs:p-2 text-xs xs:text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      {BOARD_SIZES.map(size => (
                        <option key={size} value={size}>{size}x{size}</option>
                      ))}
                    </select>
                  </div>
                  
                  {gameMode === 'ai' && (
                    <>
                      <div>
                        <label className="block text-xs sm:text-sm font-medium mb-0.5 xs:mb-1 text-gray-700">
                          Bạn chơi quân:
                        </label>
                        <div className="flex space-x-1 xs:space-x-2">
                          <button
                            onClick={() => setSettings(prev => ({ ...prev, humanColor: 'black' }))}
                            className={`flex-1 py-1.5 xs:py-2 px-2 sm:px-3 text-xs xs:text-sm rounded-lg font-medium transition-all ${
                              settings.humanColor === 'black'
                                ? 'bg-gray-800 text-white'
                                : 'bg-gray-200 hover:bg-gray-300'
                            }`}
                          >
                            ⚫ Đen
                          </button>
                          <button
                            onClick={() => setSettings(prev => ({ ...prev, humanColor: 'white' }))}
                            className={`flex-1 py-1.5 xs:py-2 px-2 sm:px-3 text-xs xs:text-sm rounded-lg font-medium transition-all ${
                              settings.humanColor === 'white'
                                ? 'bg-gray-100 text-black border-2 border-gray-800'
                                : 'bg-gray-200 hover:bg-gray-300'
                            }`}
                          >
                            ⚪ Trắng
                          </button>
                        </div>
                      </div>
                      <div>
                        <label className="block text-xs sm:text-sm font-medium mb-0.5 xs:mb-1 text-gray-700">
                          Độ khó AI:
                        </label>
                        <select
                          value={settings.difficulty}
                          onChange={(e) => setSettings(prev => ({ 
                            ...prev, 
                            difficulty: e.target.value as Difficulty 
                          }))}
                          className="w-full p-1.5 xs:p-2 text-xs xs:text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        >
                          <option value="easy">🟢 Dễ</option>
                          <option value="medium">🟡 Trung bình</option>
                          <option value="hard">🔴 Khó (MCTS)</option>
                        </select>
                      </div>
                    </>
                  )}
                </div>
              </div>

              <div className="bg-gradient-to-br from-green-50 to-green-100 p-2 xs:p-3 sm:p-4 rounded-lg border border-green-200">
                <h3 className="font-bold mb-1 xs:mb-2 sm:mb-3 text-gray-800 text-xs xs:text-sm sm:text-base">
                  📜 Lịch sử nước đi
                </h3>
                <div className="max-h-24 xs:max-h-32 sm:max-h-48 overflow-y-auto">
                  {moveHistory.slice(-10).map((move, index) => (
                    <div key={index} className="flex justify-between text-xs sm:text-sm py-0.5 xs:py-1 border-b border-green-200 last:border-0">
                      <span className="font-medium text-gray-600">
                        {moveHistory.length - 9 + index}.
                      </span>
                      <span className={`font-medium ${move.player === Stone.BLACK ? 'text-black' : 'text-gray-500'}`}>
                        {move.player === Stone.BLACK ? '⚫' : '⚪'}
                      </span>
                      <span className="font-mono text-xs">
                        {move.isPass ? 'Pass' : `${String.fromCharCode(65 + move.position.x)}${settings.boardSize - move.position.y}`}
                      </span>
                      {move.captures > 0 && (
                        <span className="text-red-600 font-bold">+{move.captures}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-2 xs:p-3 sm:p-4 rounded-lg border border-purple-200">
                <h3 className="font-bold mb-1 xs:mb-2 sm:mb-3 text-gray-800 text-xs xs:text-sm sm:text-base">
                  💾 Lưu/Tải
                </h3>
                <div className="space-y-1 xs:space-y-2">
                  <button
                    onClick={saveGame}
                    className="w-full py-1.5 xs:py-2 px-2 xs:px-3 bg-purple-600 text-white text-xs xs:text-sm rounded-lg hover:bg-purple-700 transition-colors"
                  >
                    💾 Lưu ván cờ (SGF)
                  </button>
                  <label className="block">
                    <span className="sr-only">Tải ván cờ</span>
                    <input
                      type="file"
                      accept=".sgf"
                      onChange={loadGame}
                      className="block w-full text-xs sm:text-sm file:mr-2 xs:file:mr-4 file:py-1 xs:file:py-2 file:px-2 xs:file:px-4 file:rounded-lg file:border-0 file:text-xs xs:file:text-sm file:font-semibold file:bg-purple-50 file:text-purple-700 hover:file:bg-purple-100"
                    />
                  </label>
                </div>
              </div>
            </div>

            <div className="flex-1 order-1 lg:order-2">
              <div className="flex justify-center mb-2 xs:mb-4 sm:mb-6">
                {renderBoard}
              </div>

              <div className="grid grid-cols-2 xs:grid-cols-2 sm:flex sm:justify-center gap-1 xs:gap-2 sm:gap-3 mb-2 xs:mb-4 sm:mb-6">
                <button
                  onClick={handlePass}
                  disabled={gameStatus !== 'playing'}
                  className="px-2 xs:px-3 sm:px-6 py-1.5 xs:py-2 sm:py-3 text-xs xs:text-sm sm:text-base bg-gradient-to-r from-yellow-500 to-yellow-600 text-white rounded-lg hover:from-yellow-600 hover:to-yellow-700 disabled:from-gray-300 disabled:to-gray-400 shadow-lg transition-all transform hover:scale-105"
                >
                  Pass (P)
                </button>
                <button
                  onClick={handleUndo}
                  disabled={moveHistory.length === 0}
                  className="px-2 xs:px-3 sm:px-6 py-1.5 xs:py-2 sm:py-3 text-xs xs:text-sm sm:text-base bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 disabled:from-gray-300 disabled:to-gray-400 shadow-lg transition-all transform hover:scale-105"
                >
                  ↶ Undo
                </button>
                <button
                  onClick={initializeGame}
                  className="px-2 xs:px-3 sm:px-6 py-1.5 xs:py-2 sm:py-3 text-xs xs:text-sm sm:text-base bg-gradient-to-r from-red-500 to-red-600 text-white rounded-lg hover:from-red-600 hover:to-red-700 shadow-lg transition-all transform hover:scale-105"
                >
                  🔄 Chơi lại
                </button>
                <button
                  onClick={() => {
                    const score = calculateScore();
                    setGameScore(score);
                    setShowScore(true);
                  }}
                  className="px-2 xs:px-3 sm:px-6 py-1.5 xs:py-2 sm:py-3 text-xs xs:text-sm sm:text-base bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg hover:from-green-600 hover:to-green-700 shadow-lg transition-all transform hover:scale-105"
                >
                  📊 Tính điểm
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {showTutorial && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 backdrop-blur-sm p-2 xs:p-4">
          <div className="bg-white p-4 xs:p-6 sm:p-8 rounded-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <h2 className="text-lg xs:text-xl sm:text-2xl font-bold mb-2 xs:mb-4 text-center">
              📖 Hướng dẫn chơi Cờ Vây
            </h2>
            <div className="space-y-2 xs:space-y-3 sm:space-y-4 text-xs xs:text-sm sm:text-base">
              {Object.entries(TUTORIAL_MESSAGES).map(([key, message]) => (
                <div key={key} className="p-2 xs:p-3 sm:p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-bold mb-1 xs:mb-2 capitalize">{key}:</h3>
                  <p className="text-gray-700">{message}</p>
                </div>
              ))}
            </div>
            <button
              onClick={() => setShowTutorial(false)}
              className="mt-4 xs:mt-6 w-full py-1.5 xs:py-2 sm:py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-xs xs:text-sm sm:text-base"
            >
              Đóng
            </button>
          </div>
        </div>
      )}

      {showScore && gameScore && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 backdrop-blur-sm p-2 xs:p-4">
          <div className="bg-white p-4 xs:p-6 sm:p-8 rounded-xl max-w-md w-full shadow-2xl">
            <h2 className="text-lg xs:text-xl sm:text-2xl font-bold mb-2 xs:mb-4 text-center">
              🏆 Kết quả ván cờ
            </h2>
            <div className="space-y-2 xs:space-y-3 text-xs xs:text-sm sm:text-base">
              <div className="flex justify-between p-1.5 xs:p-2 bg-gray-50 rounded">
                <span>Vùng đất Đen:</span>
                <span className="font-medium">{gameScore.blackTerritory}</span>
              </div>
              <div className="flex justify-between p-1.5 xs:p-2 bg-gray-50 rounded">
                <span>Vùng đất Trắng:</span>
                <span className="font-medium">{gameScore.whiteTerritory}</span>
              </div>
              <div className="flex justify-between p-1.5 xs:p-2 bg-gray-50 rounded">
                <span>Quân bắt Đen:</span>
                <span className="font-medium">{captures.black}</span>
              </div>
              <div className="flex justify-between p-1.5 xs:p-2 bg-gray-50 rounded">
                <span>Quân bắt Trắng:</span>
                <span className="font-medium">{captures.white}</span>
              </div>
              <div className="flex justify-between p-1.5 xs:p-2 bg-gray-50 rounded">
                <span>Komi:</span>
                <span className="font-medium">{gameScore.komi}</span>
              </div>
              <hr className="my-2 xs:my-3" />
              <div className="flex justify-between p-1.5 xs:p-2 bg-gray-800 text-white rounded">
                <span className="font-bold">Điểm Đen:</span>
                <span className="font-bold">{gameScore.blackScore.toFixed(1)}</span>
              </div>
              <div className="flex justify-between p-1.5 xs:p-2 bg-gray-100 rounded">
                <span className="font-bold">Điểm Trắng:</span>
                <span className="font-bold">{gameScore.whiteScore.toFixed(1)}</span>
              </div>
              <div className="text-center font-bold text-base xs:text-lg sm:text-xl mt-2 xs:mt-4 p-2 xs:p-3 sm:p-4 rounded-lg bg-gradient-to-r from-yellow-400 to-yellow-500 text-white">
                {gameScore.winner === 'draw' 
                  ? '🤝 HÒA!' 
                  : `🎉 ${gameScore.winner === 'black' ? 'ĐEN' : 'TRẮNG'} THẮNG!`}
              </div>
            </div>
            <div className="flex gap-2 xs:gap-3 mt-4 xs:mt-6">
              <button
                onClick={() => setShowScore(false)}
                className="flex-1 py-1.5 xs:py-2 sm:py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-all text-xs xs:text-sm sm:text-base"
              >
                Đóng
              </button>
              <button
                onClick={() => {
                  setShowScore(false);
                  initializeGame();
                }}
                className="flex-1 py-1.5 xs:py-2 sm:py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all text-xs xs:text-sm sm:text-base"
              >
                Ván mới
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GoGame;
