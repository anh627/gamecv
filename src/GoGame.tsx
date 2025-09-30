import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { produce } from 'immer';
import { useTimer } from 'react-timer-hook';
import { Stone, StoneType, GameMode, Difficulty, GameStatus, PlayerColor, Position, GameMove, Captures, GameScore, GameSettings, makeEmptyBoard, inBounds, getNeighbors, getGroupAndLiberties, tryPlay, generateStarPoints, pickAiMove, exportToSGF, importFromSGF, StoneComponent, BoardCell, TimerDisplay, Tooltip, TutorialModal, BOARD_SIZES, DEFAULT_SETTINGS, TUTORIAL_MESSAGES } from './GoGameLogic'; // Import from Part 1


// ---- Types & Constants ----
const Stone = {
  EMPTY: 0,
  BLACK: 1,
  WHITE: 2
} as const;
type StoneType = typeof Stone[keyof typeof Stone];

type GameMode = 'ai' | 'local' | 'online';
type Difficulty = 'easy' | 'medium' | 'hard';
type GameStatus = 'playing' | 'finished' | 'paused';
type PlayerColor = 'black' | 'white';

interface Position {
  x: number;
  y: number;
}

interface GameMove {
  player: StoneType;
  position: Position;
  timestamp: number;
  captures: number;
  isPass?: boolean;
  boardState: StoneType[][];
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

interface MCTSNode {
  position: Position | null;
  wins: number;
  visits: number;
  children: MCTSNode[];
  parent: MCTSNode | null;
  untriedMoves: Position[];
  playerToMove: StoneType;
}

const BOARD_SIZES = [7, 9, 13, 15, 19] as const;
const AI_CONFIG = {
  easy: { maxSimulations: 100, timeLimit: 500, simulationDepth: 30 },
  medium: { maxSimulations: 500, timeLimit: 1000, simulationDepth: 50 },
  hard: { maxSimulations: 2000, timeLimit: 2000, simulationDepth: 100 },
};

const DEFAULT_SETTINGS: GameSettings = {
  boardSize: 9,
  komi: 6.5,
  handicap: 0,
  difficulty: 'medium',
  timePerMove: 30,
  humanColor: 'black'
};

const TUTORIAL_MESSAGES = {
  placement: {
    text: "Click giao điểm trống để đặt quân. Quân cần ít nhất 1 'khí' (ô trống kề bên).",
    tip: "Với AI 'easy', đặt quân ở góc (như 3-3 trên 9x9) để dễ mở rộng."
  },
  capture: {
    text: "Bắt quân đối thủ bằng cách chặn hết 'khí' của chúng. Quân bị bắt được loại khỏi bàn.",
    tip: "Với AI 'easy', chặn khí cuối để bắt quân. AI 'medium' có thể phản công."
  },
  ko: {
    text: "Luật Ko: Không đặt quân tạo trạng thái bàn cờ lặp lại ngay sau khi bắt 1 quân.",
    tip: "Với AI 'medium', đi nước khác trước khi lấy lại quân. AI 'easy' ít tận dụng Ko."
  },
  pass: {
    text: "Pass khi không có nước đi tốt. Hai lần pass liên tiếp kết thúc ván cờ.",
    tip: "Pass sớm với AI 'easy' nếu dẫn điểm. Với AI 'medium', bảo vệ vùng đất trước."
  },
  territory: {
    text: "Vùng đất là các ô trống được bao quanh hoàn toàn bởi quân của bạn.",
    tip: "Tạo vùng đất ở góc với AI 'easy'. AI 'medium' sẽ tranh vùng đất."
  },
  scoring: {
    text: "Điểm = Vùng đất + Quân bắt được + Komi (6.5 cho trắng).",
    tip: "Bắt quân với AI 'easy'. Với AI 'medium', tạo mắt để bảo vệ vùng đất."
  },
  eyes: {
    text: "Mắt là ô trống được bao quanh bởi quân bạn. Hai mắt giúp nhóm 'sống'.",
    tip: "Tạo mắt hình vuông với AI 'medium'. Tấn công nhóm không mắt của AI 'easy'."
  }
};

// ---- Utility Functions ----
function makeEmptyBoard(size: number): StoneType[][] {
  return Array.from({ length: size }, () => Array<StoneType>(size).fill(Stone.EMPTY));
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

function getGroupAndLiberties(board: StoneType[][], x: number, y: number): {
  group: Position[];
  liberties: Set<string>;
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
  board: StoneType[][],
  x: number,
  y: number,
  color: StoneType,
  koBoard: StoneType[][] | null
): {
  legal: boolean;
  board?: StoneType[][];
  captures?: number;
  capturedPositions?: Position[];
} {
  const size = board.length;
  if (!inBounds(x, y, size) || board[y][x] !== Stone.EMPTY) return { legal: false };

  let totalCaptures = 0;
  const capturedPositions: Position[] = [];
  const opponent: StoneType = color === Stone.BLACK ? Stone.WHITE : Stone.BLACK;

  const finalBoard = produce(board, draft => {
    draft[y][x] = color;
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

  if (koBoard) {
    let isSameAsKo = true;
    for (let i = 0; i < size && isSameAsKo; i++) {
      for (let j = 0; j < size; j++) {
        if (finalBoard[i][j] !== koBoard[i][j]) {
          isSameAsKo = false;
          break;
        }
      }
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

function generateStarPoints(boardSize: number): Position[] {
  const points: Position[] = [];
  if (!BOARD_SIZES.includes(boardSize as any)) return points;

  const edge = boardSize <= 9 ? 2 : 3;
  const center = Math.floor(boardSize / 2);
  const far = boardSize - edge - 1;

  if (boardSize >= 9) {
    points.push(
      { x: edge, y: edge }, { x: edge, y: far }, { x: far, y: edge }, { x: far, y: far }
    );
  }
  if (boardSize >= 13) {
    points.push(
      { x: edge, y: center }, { x: far, y: center },
      { x: center, y: edge }, { x: center, y: far }
    );
  }
  if (boardSize >= 9) {
    points.push({ x: center, y: center });
  }

  return points;
}

// ---- Optimized MCTS AI Implementation ----
class MCTSEngine {
  private readonly config: { maxSimulations: number; timeLimit: number; simulationDepth: number };
  private readonly explorationConstant: number = Math.sqrt(2);
  private cache: Map<number, MCTSNode> = new Map();
  private readonly maxCacheSize = 1000;
  private zobristTable: number[][][] = [];

  constructor(difficulty: Difficulty, boardSize: number) {
    this.config = AI_CONFIG[difficulty];
    this.initializeZobrist(boardSize);
  }

  private initializeZobrist(size: number) {
    this.zobristTable = Array.from({ length: size }, () =>
      Array.from({ length: size }, () => [
        Math.random() * 1e9 | 0,
        Math.random() * 1e9 | 0,
        Math.random() * 1e9 | 0
      ])
    );
  }

  private getBoardKey(board: StoneType[][], player: StoneType): number {
    let hash = 0;
    for (let y = 0; y < board.length; y++) {
      for (let x = 0; x < board.length; x++) {
        if (board[y][x] !== Stone.EMPTY) {
          hash ^= this.zobristTable[y][x][board[y][x]];
        }
      }
    }
    return hash ^ (player === Stone.BLACK ? 1 : 2);
  }

  private updateCache(boardKey: number, node: MCTSNode) {
    if (this.cache.size >= this.maxCacheSize) {
      const oldestKey = this.cache.keys().next().value;
      this.cache.delete(oldestKey);
    }
    this.cache.set(boardKey, node);
  }

  selectMove(board: StoneType[][], color: StoneType, koBoard: StoneType[][] | null): Position | null {
    const boardKey = this.getBoardKey(board, color);
    let root = this.cache.get(boardKey);
    if (!root) {
      root = this.createNode(null, null, color, board);
      this.updateCache(boardKey, root);
    }

    const startTime = performance.now();
    let simulations = 0;

    while (performance.now() - startTime < this.config.timeLimit && simulations < this.config.maxSimulations) {
      const node = this.select(root, board, koBoard);
      const result = this.simulate(node, board, color, koBoard);
      this.backpropagate(node, result);
      simulations++;
    }

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
    playerToMove: StoneType,
    board: StoneType[][]
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

  private getPossibleMoves(board: StoneType[][], color: StoneType): Position[] {
    const size = board.length;
    const moves: Position[] = [];
    const priorityMoves: Position[] = [];
    const regularMoves: Position[] = [];

    const relevantPositions = getRelevantPositions(board, 2);

    for (const pos of relevantPositions) {
      const result = tryPlay(board, pos.x, pos.y, color, null);
      if (result.legal) {
        const neighbors = getNeighbors(pos.x, pos.y, size);
        const hasStoneNearby = neighbors.some(n => board[n.y][n.x] !== Stone.EMPTY);
        if (hasStoneNearby || result.captures! > 0) {
          priorityMoves.push(pos);
        } else {
          regularMoves.push(pos);
        }
      }
    }

    if (priorityMoves.length === 0 && regularMoves.length === 0) {
      return [{ x: -1, y: -1 }];
    }

    return [...priorityMoves, ...regularMoves];
  }

  private select(node: MCTSNode, board: StoneType[][], koBoard: StoneType[][] | null): MCTSNode {
    let current = node;
    let currentBoard = board.map(row => [...row]);

    while (current.untriedMoves.length === 0 && current.children.length > 0) {
      current = this.selectBestChild(current);
      if (current.position && current.position.x !== -1) {
        const result = tryPlay(currentBoard, current.position.x, current.position.y,
          current.parent!.playerToMove, koBoard);
        if (result.board) {
          currentBoard = result.board;
        }
      }
    }

    if (current.untriedMoves.length > 0) {
      const topMoves = current.untriedMoves
        .map(pos => ({
          pos,
          score: calculateMoveScore(currentBoard, pos.x, pos.y, current.playerToMove, 'medium', koBoard)
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, Math.min(10, current.untriedMoves.length));

      const move = topMoves[Math.floor(Math.random() * topMoves.length)].pos;
      const moveIndex = current.untriedMoves.findIndex(p => p.x === move.x && p.y === move.y);
      current.untriedMoves.splice(moveIndex, 1);

      const nextPlayer = current.playerToMove === Stone.BLACK ? Stone.WHITE : Stone.BLACK;
      const child = this.createNode(current, move, nextPlayer, currentBoard);
      current.children.push(child);

      const childBoard = tryPlay(currentBoard, move.x, move.y, current.playerToMove, koBoard).board;
      if (childBoard) {
        this.updateCache(this.getBoardKey(childBoard, nextPlayer), child);
      }
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

  private simulate(node: MCTSNode, board: StoneType[][], originalColor: StoneType, koBoard: StoneType[][] | null): number {
    let simulationBoard = board.map(row => [...row]);
    let currentPlayer = node.playerToMove;
    let passes = 0;
    let moveCount = 0;

    while (moveCount < this.config.simulationDepth && passes < 2) {
      const moves = this.getPossibleMoves(simulationBoard, currentPlayer);

      if (moves.length === 0 || moves[0].x === -1) {
        passes++;
      } else {
        passes = 0;
        const topMoves = moves.slice(0, Math.min(5, moves.length));
        const move = topMoves[Math.floor(Math.random() * topMoves.length)];
        const result = tryPlay(simulationBoard, move.x, move.y, currentPlayer, koBoard);
        if (result.board) {
          simulationBoard = result.board;
        }
      }

      currentPlayer = currentPlayer === Stone.BLACK ? Stone.WHITE : Stone.BLACK;
      moveCount++;
    }

    const score = this.evaluateBoard(simulationBoard, originalColor);
    return score > 0 ? 1 : 0;
  }

  private evaluateBoard(board: StoneType[][], color: StoneType): number {
    let myStones = 0;
    let oppStones = 0;
    let myTerritory = 0;
    let oppTerritory = 0;
    const opponent = color === Stone.BLACK ? Stone.WHITE : Stone.BLACK;
    const size = board.length;

    const visited = new Set<string>();

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const stone = board[y][x];
        const key = `${x},${y}`;

        if (visited.has(key)) continue;

        if (stone === color) {
          myStones++;
        } else if (stone === opponent) {
          oppStones++;
        } else {
          const { group, liberties } = getGroupAndLiberties(board, x, y);
          group.forEach(pos => visited.add(`${pos.x},${pos.y}`));

          let myNeighbors = 0;
          let oppNeighbors = 0;
          for (const pos of group) {
            const neighbors = getNeighbors(pos.x, pos.y, size);
            for (const n of neighbors) {
              if (board[n.y][n.x] === color) myNeighbors++;
              else if (board[n.y][n.x] === opponent) oppNeighbors++;
            }
          }

          const isEye = liberties.size === group.length && group.length <= 2;
          if (myNeighbors > oppNeighbors && myNeighbors >= 3) {
            myTerritory += group.length + (isEye ? 5 : 0);
          } else if (oppNeighbors > myNeighbors && oppNeighbors >= 3) {
            oppTerritory += group.length + (isEye ? 5 : 0);
          }
        }
      }
    }

    const myGroups = this.countGroups(board, color);
    const oppGroups = this.countGroups(board, opponent);
    const groupBonus = (myGroups - oppGroups) * 2;

    return (myStones + myTerritory + groupBonus) - (oppStones + oppTerritory);
  }

  private countGroups(board: StoneType[][], color: StoneType): number {
    const size = board.length;
    const visited = new Set<string>();
    let groupCount = 0;

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (board[y][x] === color && !visited.has(`${x},${y}`)) {
          const { group } = getGroupAndLiberties(board, x, y);
          group.forEach(pos => visited.add(`${pos.x},${pos.y}`));
          groupCount++;
        }
      }
    }

    return groupCount;
  }

  private backpropagate(node: MCTSNode | null, result: number): void {
    while (node !== null) {
      node.visits++;
      node.wins += node.playerToMove === node.parent?.playerToMove ? 1 - result : result;
      node = node.parent;
    }
  }
}

// ---- Fast Heuristic AI for easy/medium ----
function getRelevantPositions(board: StoneType[][], maxDistance: number = 2): Position[] {
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
    const center = Math.floor(size / 2);
    const positions = [
      { x: center, y: center },
      { x: 3, y: 3 }, { x: size - 4, y: 3 },
      { x: 3, y: size - 4 }, { x: size - 4, y: size - 4 }
    ];

    for (const pos of positions) {
      if (inBounds(pos.x, pos.y, size) && board[pos.y][pos.x] === Stone.EMPTY) {
        relevantPositions.add(`${pos.x},${pos.y}`);
      }
    }
  }

  return Array.from(relevantPositions).map(key => {
    const [x, y] = key.split(',').map(Number);
    return { x, y };
  });
}

function calculateMoveScore(
  board: StoneType[][],
  x: number,
  y: number,
  color: StoneType,
  difficulty: Difficulty,
  koBoard: StoneType[][] | null
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
  score -= distanceFromCenter * 2;

  const neighbors = getNeighbors(x, y, size);
  const friendlyNeighbors = neighbors.filter(n => board[n.y][n.x] === color).length;
  const opponentNeighbors = neighbors.filter(n =>
    board[n.y][n.x] === (color === Stone.BLACK ? Stone.WHITE : Stone.BLACK)).length;

  score += friendlyNeighbors * 10;
  score += opponentNeighbors * 8;

  if (difficulty === 'easy') {
    score += (x === 3 || x === size - 4 || y === 3 || y === size - 4) ? 15 : 0;
  } else if (difficulty === 'medium') {
    const cornerBonus = (x === 0 || x === size - 1) && (y === 0 || y === size - 1) ? 10 : 0;
    const sideBonus = (x === 0 || x === size - 1 || y === 0 || y === size - 1) ? 5 : 0;
    const { group, liberties } = getGroupAndLiberties(board, x, y);
    const eyePotential = liberties.size === 1 && friendlyNeighbors >= 3 ? 15 : 0;

    score += cornerBonus + sideBonus + eyePotential;

    if (result.board) {
      const opponent = color === Stone.BLACK ? Stone.WHITE : Stone.BLACK;
      const opponentMoves = getRelevantPositions(result.board, 2);
      let maxOpponentScore = -Infinity;
      for (const oppPos of opponentMoves.slice(0, 5)) {
        const oppScore = calculateMoveScore(result.board, oppPos.x, oppPos.y, opponent, 'easy', null);
        maxOpponentScore = Math.max(maxOpponentScore, oppScore);
      }
      score -= maxOpponentScore * 0.5;
    }
  }

  return score;
}

function pickAiMove(
  board: StoneType[][],
  color: StoneType,
  difficulty: Difficulty,
  koBoard: StoneType[][] | null
): Position {
  if (!board?.length || !board[0]?.length) {
    throw new Error('Invalid board: Board must be a non-empty 2D array');
  }
  if (koBoard && (koBoard.length !== board.length || koBoard[0].length !== board[0].length)) {
    throw new Error('Invalid koBoard: Size must match board');
  }

  if (difficulty === 'hard') {
    const mcts = new MCTSEngine(difficulty, board.length);
    return mcts.selectMove(board, color, koBoard) || { x: -1, y: -1 };
  }

  const maxDistance = difficulty === 'medium' ? 4 : 2;
  const relevantPositions = getRelevantPositions(board, maxDistance);
  const candidates: { position: Position; score: number }[] = [];

  for (const pos of relevantPositions) {
    const score = calculateMoveScore(board, pos.x, pos.y, color, difficulty, koBoard);
    if (score > -Infinity) {
      candidates.push({ position: pos, score });
    }
  }

  if (candidates.length === 0) {
    return { x: -1, y: -1 };
  }

  candidates.sort((a, b) => b.score - a.score);
  let selectedIndex = 0;
  if (difficulty === 'easy' && candidates.length > 3) {
    selectedIndex = Math.floor(Math.random() * Math.min(3, candidates.length));
  } else if (difficulty === 'medium' && candidates.length > 2) {
    selectedIndex = Math.floor(Math.random() * Math.min(2, candidates.length));
  }

  return candidates[selectedIndex].position;
}

// ---- Enhanced SGF Functions ----
function positionToSGF(pos: Position): string {
  if (pos.x === -1 && pos.y === -1) return '';
  const x = String.fromCharCode(97 + pos.x);
  const y = String.fromCharCode(97 + pos.y);
  return x + y;
}

function sgfToPosition(sgf: string): Position {
  if (sgf === '') return { x: -1, y: -1 };
  if (sgf.length !== 2) throw new Error(`Invalid SGF position: ${sgf}`);
  const x = sgf.charCodeAt(0) - 97;
  const y = sgf.charCodeAt(1) - 97;
  if (x < 0 || y < 0) throw new Error(`Invalid coordinates in SGF: ${sgf}`);
  return { x, y };
}

function exportToSGF(
  moves: GameMove[],
  settings: GameSettings,
  score?: GameScore
): string {
  let sgf = `(;FF[4]GM[1]SZ[${settings.boardSize}]`;
  sgf += `KM[${settings.komi}]`;

  if (score) {
    const result = score.winner === 'black'
      ? `B+${(score.blackScore - score.whiteScore).toFixed(1)}`
      : `W+${(score.whiteScore - score.blackScore).toFixed(1)}`;
    sgf += `RE[${result}]`;
  }

  for (const move of moves) {
    const color = move.player === Stone.BLACK ? 'B' : 'W';
    const coord = positionToSGF(move.position);
    sgf += `;${color}[${coord}]`;
  }

  sgf += ')';
  return sgf;
}

function importFromSGF(sgfContent: string): {
  moves: { position: Position; player: StoneType }[];
  boardSize: number;
  komi: number;
} | null {
  try {
    const content = sgfContent.replace(/\s+/g, ' ').trim();
    const sizeMatch = content.match(/SZ\[(\d+)\]/);
    const boardSize = sizeMatch ? parseInt(sizeMatch[1], 10) : 19;
    const komiMatch = content.match(/KM\[([0-9.-]+)\]/);
    const komi = komiMatch ? parseFloat(komiMatch[1]) : 6.5;
    const moves: { position: Position; player: StoneType }[] = [];
    const movePattern = /;([BW])\[([a-z]{0,2})\]/g;
    let match;

    while ((match = movePattern.exec(content)) !== null) {
      const player = match[1] === 'B' ? Stone.BLACK : Stone.WHITE;
      const position = sgfToPosition(match[2]);
      moves.push({ position, player });
    }

    return { moves, boardSize, komi };
  } catch (error) {
    console.error('SGF parse error:', error);
    alert('Không thể phân tích tệp SGF. Vui lòng kiểm tra định dạng và thử lại.');
    return null;
  }
}

// ---- UI Components ----
const StoneComponent = React.memo(({ stone, size, isLastMove, isAnimating }: {
  stone: StoneType;
  size: 'small' | 'medium' | 'large';
  isLastMove: boolean;
  isAnimating: boolean;
}) => {
  const sizeClasses = {
    small: 'w-3 h-3 xs:w-2.5 xs:h-2.5 sm:w-4 sm:h-4',
    medium: 'w-5 h-5 xs:w-4 xs:h-4 sm:w-7 sm:h-7',
    large: 'w-8 h-8 xs:w-6 xs:h-6 sm:w-10 sm:h-10'
  };

  if (stone === Stone.EMPTY) return null;

  return (
    <div
      className={`
        rounded-full transition-all duration-300 relative 
        ${sizeClasses[size]} 
        ${stone === Stone.BLACK 
          ? 'bg-gradient-to-br from-gray-700 via-gray-800 to-black shadow-stone-black' 
          : 'bg-gradient-to-br from-gray-100 via-white to-gray-200 shadow-stone-white'} 
        ${isLastMove ? 'ring-2 ring-red-500 ring-opacity-60 animate-pulse' : ''} 
        ${isAnimating ? 'animate-place-stone scale-110' : ''}
      `}
      style={{
        boxShadow: stone === Stone.BLACK
          ? '2px 2px 4px rgba(0,0,0,0.5), inset -1px -1px 2px rgba(255,255,255,0.1)'
          : '2px 2px 4px rgba(0,0,0,0.3), inset -1px -1px 2px rgba(0,0,0,0.1)'
      }}
    >
      <div
        className={`absolute top-0.5 left-0.5 xs:top-0.5 xs:left-0.5 sm:top-1 sm:left-1 w-1.5 h-1.5 xs:w-1 xs:h-1 sm:w-2 sm:h-2 rounded-full ${
          stone === Stone.BLACK ? 'bg-gray-600 opacity-30' : 'bg-white opacity-60'
        }`}
      />
    </div>
  );
});

const BoardCell = React.memo(({
  position,
  stone,
  cellSize,
  isHovered,
  isValidMove,
  isStarPoint,
  isLastMove,
  onHover,
  onClick,
  currentPlayer,
  isAnimating
}: {
  position: Position;
  stone: StoneType;
  cellSize: number;
  isHovered: boolean;
  isValidMove: boolean;
  isStarPoint: boolean;
  isLastMove: boolean;
  onHover: (position: Position | null) => void;
  onClick: (position: Position) => void;
  currentPlayer: StoneType;
  isAnimating: boolean;
}) => {
  const handleMouseEnter = () => onHover(position);
  const handleMouseLeave = () => onHover(null);
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      onClick(position);
    }
  };

  return (
    <div
      style={{
        width: cellSize,
        height: cellSize,
        background: isStarPoint ? 'rgba(0, 0, 0, 0.3)' : 'transparent',
        border: isValidMove && isHovered ? '2px solid yellow' : '1px solid #8B4513', // Brown for wood theme
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'border 0.2s ease-in-out',
      }}
      className={isValidMove && isHovered ? 'shadow-md' : ''}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={() => onClick(position)}
      onTouchStart={handleMouseEnter}
      onKeyDown={handleKeyDown}
      role="gridcell"
      aria-label={`Vị trí ${position.x},${position.y}, ${
        stone === Stone.BLACK ? 'Quân đen' : stone === Stone.WHITE ? 'Quân trắng' : 'Trống'
      }${isLastMove ? ', nước đi cuối' : ''}`}
      tabIndex={0}
    >
      {stone !== Stone.EMPTY && (
        <StoneComponent
          stone={stone}
          size="medium"
          isLastMove={isLastMove}
          isAnimating={isAnimating}
        />
      )}
      {isHovered && isValidMove && (
        <Tooltip
          message={TUTORIAL_MESSAGES.placement.text}
          position={position}
          cellSize={cellSize}
          boardSize={9} // Replace with actual from settings
        />
      )}
    </div>
  );
});

const TimerDisplay = React.memo(({
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
    <div className={`px-3 py-1.5 rounded-lg shadow-md ${isActive ? 'bg-red-600 text-white' : 'bg-gray-200 text-gray-800'} font-semibold`}>
      {color === 'black' ? '⚫' : '⚪'} {minutes}:{seconds.toString().padStart(2, '0')}
    </div>
  );
});

const Tooltip = React.memo(({ message, position, cellSize, boardSize }: {
  message: string;
  position: Position;
  cellSize: number;
  boardSize: number;
}) => {
  const left = position.x * cellSize + cellSize / 2;
  const top = position.y * cellSize - 40;
  const adjustedLeft = Math.min(Math.max(left, 50), boardSize * cellSize - 50);
  const adjustedTop = position.y < 2 ? position.y * cellSize + cellSize + 10 : top;

  return (
    <div
      style={{
        position: 'absolute',
        left: adjustedLeft,
        top: adjustedTop,
        transform: 'translateX(-50%)',
        background: 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        padding: '8px 12px',
        borderRadius: '8px',
        fontSize: '14px',
        fontFamily: 'Arial, sans-serif',
        maxWidth: '250px',
        textAlign: 'center',
        zIndex: 1000,
        boxShadow: '0 4px 8px rgba(0,0,0,0.3)',
        border: '1px solid rgba(255,255,255,0.2)',
      }}
    >
      {message}
    </div>
  );
});

const TutorialModal = React.memo(({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const steps = Object.entries(TUTORIAL_MESSAGES).map(([key, { text, tip }]) => ({
    key: key.charAt(0).toUpperCase() + key.slice(1),
    text,
    tip,
  }));

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
    >
      <div
        className="bg-gradient-to-b from-white to-gray-100 p-6 rounded-xl shadow-2xl max-w-md w-full text-center transform transition-all duration-300 scale-100"
      >
        <h2 className="text-2xl font-bold mb-4 text-gray-800">Hướng dẫn chơi cờ vây</h2>
        <h3 className="text-xl font-semibold mb-2 text-gray-700">{steps[currentStep].key}</h3>
        <p className="text-gray-600 mb-4">{steps[currentStep].text}</p>
        <p className="text-gray-500"><strong>Mẹo:</strong> {steps[currentStep].tip}</p>
        <div className="flex justify-around mt-6">
          <button
            onClick={() => setCurrentStep((prev) => Math.max(0, prev - 1))}
            disabled={currentStep === 0}
            className={`px-4 py-2 rounded-lg text-white ${currentStep === 0 ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-500 hover:bg-blue-600'}`}
          >
            Trước
          </button>
          <button
            onClick={() => setCurrentStep((prev) => Math.min(steps.length - 1, prev + 1))}
            disabled={currentStep === steps.length - 1}
            className={`px-4 py-2 rounded-lg text-white ${currentStep === steps.length - 1 ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-500 hover:bg-blue-600'}`}
          >
            Tiếp
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-red-500 text-white hover:bg-red-600"
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
});

const GoGame: React.FC = () => {
  // State declarations
  const [animatingCaptures, setAnimatingCaptures] = useState<Position[]>([]);
  const [gameMode, setGameMode] = useState<GameMode>('local');
  const [settings, setSettings] = useState<GameSettings>(DEFAULT_SETTINGS);
  const [board, setBoard] = useState<StoneType[][]>(() => makeEmptyBoard(settings.boardSize));
  const [currentPlayer, setCurrentPlayer] = useState<StoneType>(Stone.BLACK);
  const [captures, setCaptures] = useState<Captures>({ black: 0, white: 0 });
  const [moveHistory, setMoveHistory] = useState<GameMove[]>([]);
  const [gameStatus, setGameStatus] = useState<GameStatus>('playing');
  const [passCount, setPassCount] = useState(0);
  const [hoverPosition, setHoverPosition] = useState<Position | null>(null);
  const [lastMove, setLastMove] = useState<Position | null>(null);
  const [showScore, setShowScore] = useState(false);
  const [gameScore, setGameScore] = useState<GameScore | null>(null);
  const [timerExpiry, setTimerExpiry] = useState<Date>(() => {
    const now = new Date();
    now.setSeconds(now.getSeconds() + settings.timePerMove);
    return now;
  });
  const [isTimerActive, setIsTimerActive] = useState(true);
  const [showTutorial, setShowTutorial] = useState(false);
  const [isLoadingAI, setIsLoadingAI] = useState(false);
  const koHistoryRef = useRef<StoneType[][] | null>(null);
  const aiMoveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const boardHistoryRef = useRef<StoneType[][][]>([]);

  const getAIColor = useCallback((): StoneType => {
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
    setTimerExpiry(() => {
      const now = new Date();
      now.setSeconds(now.getSeconds() + settings.timePerMove);
      return now;
    });
    koHistoryRef.current = null;
    boardHistoryRef.current = [];
    if (gameMode === 'ai' && settings.humanColor === 'white') {
      setTimeout(() => handleAIMove(), 500);
    }
  }, [settings.boardSize, settings.humanColor, settings.timePerMove, gameMode]);

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
    if (!inBounds(x, y, board.length) || board[y][x] !== Stone.EMPTY || gameStatus !== 'playing') {
      return false;
    }
    if (gameMode === 'ai' && currentPlayer === getAIColor()) {
      return false;
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

    const floodFill = (startX: number, startY: number): { points: Position[]; owner: StoneType; isSeki: boolean } => {
      if (visited.has(`${startX},${startY}`) || board[startY][startX] !== Stone.EMPTY) {
        return { points: [], owner: Stone.EMPTY, isSeki: false };
      }
      const queue: Position[] = [{ x: startX, y: startY }];
      const territory: Position[] = [];
      const borderStones = new Set<StoneType>();
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
      let owner: StoneType = Stone.EMPTY;
      let isSeki = false;
      if (borderStones.size === 2 && borderGroups.size >= 2) {
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
    const winner = Math.abs(blackScore - whiteScore) < 0.1 ? 'draw' : blackScore > whiteScore ? 'black' : 'white';
    return { blackScore, whiteScore, blackTerritory: territory.black, whiteTerritory: territory.white, komi: settings.komi, winner };
  }, [board, captures, settings.komi, calculateTerritory]);

  const handleAIMove = useCallback(() => {
    if (gameStatus !== 'playing' || currentPlayer !== getAIColor()) return;
    setIsLoadingAI(true);
    aiMoveTimeoutRef.current = setTimeout(() => {
      try {
        const aiMove = pickAiMove(board, currentPlayer, settings.difficulty, koHistoryRef.current);
        setIsLoadingAI(false);
        if (aiMove && aiMove.x !== -1) {
          handlePlaceStone(aiMove);
        } else {
          handlePass();
        }
      } catch (error) {
        console.error('AI move error:', error);
        setIsLoadingAI(false);
        alert('Lỗi khi AI tính toán nước đi. Vui lòng thử lại.');
      }
    }, 100);
  }, [board, currentPlayer, gameStatus, settings.difficulty, getAIColor]);

  const handlePlaceStone = useCallback((position: Position) => {
    if (!isValidMove(position) && !(gameMode === 'ai' && currentPlayer === getAIColor())) return;
    const result = tryPlay(board, position.x, position.y, currentPlayer, koHistoryRef.current);
    if (!result.legal || !result.board) return;

    boardHistoryRef.current = [...boardHistoryRef.current, board];
    koHistoryRef.current = result.captures === 1 ? board : null;
    setBoard(result.board);
    setLastMove(position);
    if (result.captures && result.captures > 0) {
      setCaptures(prev => ({
        ...prev,
        [currentPlayer === Stone.BLACK ? 'black' : 'white']: prev[currentPlayer === Stone.BLACK ? 'black' : 'white'] + result.captures!
      }));
      if (result.capturedPositions) {
        setAnimatingCaptures(result.capturedPositions);
        animationFrameRef.current = requestAnimationFrame(() => {
          setTimeout(() => setAnimatingCaptures([]), 600);
        });
      }
    }
    const newMove: GameMove = {
      player: currentPlayer,
      position,
      timestamp: Date.now(),
      captures: result.captures || 0,
      boardState: produce(result.board, draft => draft)
    };
    setMoveHistory(prev => [...prev, newMove]);
    const nextPlayer: StoneType = currentPlayer === Stone.BLACK ? Stone.WHITE : Stone.BLACK;
    setCurrentPlayer(nextPlayer);
    setPassCount(0);
    setTimerExpiry(() => {
      const now = new Date();
      now.setSeconds(now.getSeconds() + settings.timePerMove);
      return now;
    });
    if (gameMode === 'ai' && nextPlayer === getAIColor()) {
      handleAIMove();
    }
  }, [board, currentPlayer, gameMode, isValidMove, settings.timePerMove, getAIColor, handleAIMove]);

  const handlePass = useCallback(() => {
    const newPassCount = passCount + 1;
    setPassCount(newPassCount);
    const passMove: GameMove = {
      player: currentPlayer,
      position: { x: -1, y: -1 },
      timestamp: Date.now(),
      captures: 0,
      isPass: true,
      boardState: produce(board, draft => draft)
    };
    setMoveHistory(prev => [...prev, passMove]);
    if (newPassCount >= 2) {
      setGameStatus('finished');
      const finalScore = calculateScore();
      setGameScore(finalScore);
      setShowScore(true);
      setIsTimerActive(false);
    } else {
      const nextPlayer: StoneType = currentPlayer === Stone.BLACK ? Stone.WHITE : Stone.BLACK;
      setCurrentPlayer(nextPlayer);
      setTimerExpiry(() => {
        const now = new Date();
        now.setSeconds(now.getSeconds() + settings.timePerMove);
        return now;
      });
      if (gameMode === 'ai' && nextPlayer === getAIColor()) {
        handleAIMove();
      }
    }
  }, [passCount, currentPlayer, board, gameMode, calculateScore, settings.timePerMove, getAIColor, handleAIMove]);

  const handleUndo = useCallback(() => {
    if (moveHistory.length === 0) return;
    let movesToUndo = gameMode === 'ai' && moveHistory[moveHistory.length - 1].player === getAIColor() ? 1 : 2;
    movesToUndo = Math.min(movesToUndo, moveHistory.length);
    const newHistory = moveHistory.slice(0, -movesToUndo);
    setMoveHistory(newHistory);
    const previousBoard = boardHistoryRef.current[boardHistoryRef.current.length - movesToUndo] || makeEmptyBoard(settings.boardSize);
    setBoard(previousBoard);
    setLastMove(newHistory.length > 0 && !newHistory[newHistory.length - 1].isPass ? newHistory[newHistory.length - 1].position : null);
    let blackCaptures = 0;
    let whiteCaptures = 0;
    for (const move of newHistory) {
      if (move.player === Stone.BLACK) blackCaptures += move.captures;
      else whiteCaptures += move.captures;
    }
    setCaptures({ black: blackCaptures, white: whiteCaptures });
    setCurrentPlayer(newHistory.length > 0 ? (newHistory[newHistory.length - 1].player === Stone.BLACK ? Stone.WHITE : Stone.BLACK) : Stone.BLACK);
    setPassCount(newHistory.slice(-2).filter(m => m.isPass).length);
    setGameStatus('playing');
    setShowScore(false);
    koHistoryRef.current = null;
    boardHistoryRef.current = boardHistoryRef.current.slice(0, -movesToUndo);
    setTimerExpiry(() => {
      const now = new Date();
      now.setSeconds(now.getSeconds() + settings.timePerMove);
      return now;
    });
  }, [moveHistory, gameMode, getAIColor, settings.boardSize, settings.timePerMove]);

  const saveGame = useCallback(() => {
    try {
      const sgf = exportToSGF(moveHistory, settings, gameScore || undefined);
      const blob = new Blob([sgf], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `go_game_${new Date().toISOString().replace(/[:.]/g, '-')}.sgf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Save game error:', error);
      alert('Lỗi khi lưu ván cờ. Vui lòng thử lại.');
    }
  }, [moveHistory, settings, gameScore]);

  const loadGame = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (e) => {
      const content = e.target?.result as string;
      const gameData = importFromSGF(content);
      if (!gameData) return;
      try {
        setSettings(prev => ({ ...prev, boardSize: gameData.boardSize, komi: gameData.komi }));
        const newBoard = makeEmptyBoard(gameData.boardSize);
        let currentBoard = newBoard;
        let currentCaptures = { black: 0, white: 0 };
        let koBoard: StoneType[][] | null = null;
        const newMoveHistory: GameMove[] = [];
        for (const move of gameData.moves) {
          if (move.position.x === -1 && move.position.y === -1) {
            const passMove: GameMove = {
              player: move.player,
              position: move.position,
              timestamp: Date.now(),
              captures: 0,
              isPass: true,
              boardState: produce(currentBoard, draft => draft)
            };
            newMoveHistory.push(passMove);
          } else {
            const result = tryPlay(currentBoard, move.position.x, move.position.y, move.player, koBoard);
            if (!result.legal || !result.board) continue;
            currentBoard = result.board;
            if (result.captures && result.captures > 0) {
              currentCaptures[move.player === Stone.BLACK ? 'black' : 'white'] += result.captures;
            }
            koBoard = result.captures === 1 ? currentBoard : null;
            const gameMove: GameMove = {
              player: move.player,
              position: move.position,
              timestamp: Date.now(),
              captures: result.captures || 0,
              boardState: produce(currentBoard, draft => draft)
            };
            newMoveHistory.push(gameMove);
            boardHistoryRef.current.push(currentBoard);
          }
        }
        setBoard(currentBoard);
        setCaptures(currentCaptures);
        setMoveHistory(newMoveHistory);
        const lastMove = newMoveHistory[newMoveHistory.length - 1];
        setCurrentPlayer(lastMove ? (lastMove.player === Stone.BLACK ? Stone.WHITE : Stone.BLACK) : Stone.BLACK);
        setLastMove(lastMove && !lastMove.isPass ? lastMove.position : null);
        setPassCount(newMoveHistory.slice(-2).filter(m => m.isPass).length);
        setGameStatus('playing');
        setShowScore(false);
        koHistoryRef.current = koBoard;
        event.target.value = '';
      } catch (error) {
        console.error('Load game error:', error);
        alert('Lỗi khi tải ván cờ. Vui lòng kiểm tra định dạng SGF.');
      }
    };
    reader.onerror = () => alert('Lỗi khi đọc file. Vui lòng thử lại.');
    reader.readAsText(file);
  }, []);

  useEffect(() => {
    initializeGame();
    return cleanup;
  }, [initializeGame, cleanup]);

  useEffect(() => {
    if (gameMode === 'ai' && currentPlayer === getAIColor() && gameStatus === 'playing') {
      handleAIMove();
    }
  }, [gameMode, currentPlayer, gameStatus, getAIColor, handleAIMove]);

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === 'p' || e.key === 'P') handlePass();
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        handleUndo();
      }
      if (e.key === 't' || e.key === 'T') setShowTutorial(true);
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [handlePass, handleUndo]);

  const renderBoard = useMemo(() => {
    const size = settings.boardSize;
    const windowWidth = typeof window !== 'undefined' ? window.innerWidth : 800;
    const maxDisplaySize = Math.min(windowWidth - 40, size <= 9 ? 400 : size <= 13 ? 500 : 600);
    const cellSize = maxDisplaySize / (size - 1);
    const padding = cellSize / 2;
    const starPoints = generateStarPoints(size);

    return (
      <div
        className="relative rounded-xl shadow-2xl mx-auto transform transition-transform duration-300 hover:scale-[1.01]"
        style={{
          width: maxDisplaySize + padding * 2,
          height: maxDisplaySize + padding * 2,
          background: 'linear-gradient(135deg, #d2b48c 0%, #e6c088 50%, #d2b48c 100%)',
          padding,
          boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
        }}
      >
        <div
          className="absolute inset-0 opacity-30 rounded-xl pointer-events-none"
          style={{
            backgroundImage: `url('https://images.unsplash.com/photo-1602529843548-c1d5d95d7c99?auto=format&fit=crop&w=500&q=60')`,
            backgroundSize: 'cover',
            filter: 'sepia(0.3)',
          }}
        />
        <svg
          className="absolute inset-0 pointer-events-none"
          style={{ width: maxDisplaySize + padding * 2, height: maxDisplaySize + padding * 2 }}
        >
          {Array.from({ length: size }).map((_, index) => (
            <line
              key={`h-${index}`}
              x1={padding}
              y1={padding + index * cellSize}
              x2={padding + maxDisplaySize}
              y2={padding + index * cellSize}
              stroke="#5c4033"
              strokeWidth="1.5"
              opacity="0.9"
            />
          ))}
          {Array.from({ length: size }).map((_, index) => (
            <line
              key={`v-${index}`}
              x1={padding + index * cellSize}
              y1={padding}
              x2={padding + index * cellSize}
              y2={padding + maxDisplaySize}
              stroke="#5c4033"
              strokeWidth="1.5"
              opacity="0.9"
            />
          ))}
          {starPoints.map((point, index) => (
            <circle
              key={`star-${index}`}
              cx={padding + point.x * cellSize}
              cy={padding + point.y * cellSize}
              r={size >= 19 ? 4 : size >= 13 ? 3 : 2.5}
              fill="#5c4033"
            />
          ))}
        </svg>
        {board.map((row, y) =>
          row.map((stone, x) => {
            const isLastMovePosition = lastMove?.x === x && lastMove?.y === y;
            const isAnimating = animatingCaptures.some(pos => pos.x === x && pos.y === y);
            return (
              <BoardCell
                key={`cell-${x}-${y}`}
                position={{ x, y }}
                stone={stone}
                boardSize={size}
                cellSize={cellSize}
                isHovered={hoverPosition?.x === x && hoverPosition?.y === y}
                isValidMove={isValidMove({ x, y })}
                isStarPoint={false}
                isLastMove={isLastMovePosition}
                onHover={setHoverPosition}
                onClick={handlePlaceStone}
                currentPlayer={currentPlayer}
                isAnimating={isAnimating}
              />
            );
          })
        )}
        {isLoadingAI && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-20 rounded-xl">
            <div className="bg-white px-4 py-2 rounded-lg shadow-lg animate-pulse flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm font-medium text-gray-800">AI đang tính toán...</span>
            </div>
          </div>
        )}
      </div>
    );
  }, [board, settings.boardSize, hoverPosition, lastMove, currentPlayer, isValidMove, handlePlaceStone, isLoadingAI, animatingCaptures]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 via-gray-200 to-gray-300 py-4 sm:py-8 px-2 sm:px-6">
      <style>
        {`
          @keyframes place-stone {
            0% { transform: scale(0.5); opacity: 0.5; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(1); opacity: 1; }
          }
          @keyframes capture-stone {
            0% { transform: scale(1); opacity: 1; }
            100% { transform: scale(0.5); opacity: 0; }
          }
          .animate-capture-stone {
            animation: capture-stone 0.6s ease-out forwards;
          }
          .shadow-stone-black {
            box-shadow: 2px 2px 6px rgba(0,0,0,0.5), inset -1px -1px 2px rgba(255,255,255,0.2);
          }
          .shadow-stone-white {
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3), inset -1px -1px 2px rgba(0,0,0,0.1);
          }
        `}
      </style>
      <div className="max-w-7xl mx-auto">
        <div className="bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 text-white p-4 sm:p-6 rounded-t-2xl shadow-xl">
          <div className="flex flex-col sm:flex-row justify-between items-center gap-3 sm:gap-4">
            <h1 className="text-xl sm:text-3xl font-extrabold tracking-tight">⚫⚪ Cờ Vây Pro</h1>
            <div className="flex items-center space-x-3 sm:space-x-4">
              <span className="flex items-center gap-2">
                <div
                  className={`w-6 h-6 sm:w-8 sm:h-8 rounded-full transition-all duration-300 ${
                    currentPlayer === Stone.BLACK
                      ? 'bg-gradient-to-br from-gray-800 to-black shadow-stone-black'
                      : 'bg-gradient-to-br from-white to-gray-200 shadow-stone-white'
                  }`}
                />
                <span className="text-sm sm:text-lg font-medium">
                  Lượt: {currentPlayer === Stone.BLACK ? 'Đen' : 'Trắng'}
                </span>
              </span>
              <button
                onClick={() => setShowTutorial(true)}
                className="p-2 bg-blue-600 rounded-full hover:bg-blue-700 transition-colors duration-200 shadow-md"
                aria-label="Hướng dẫn"
              >
                <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" stroke="white" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
                </svg>
              </button>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-b-2xl shadow-xl p-4 sm:p-8">
          <div className="mb-4 sm:mb-6 flex flex-col sm:flex-row justify-center gap-2 sm:gap-4">
            <button
              onClick={() => setGameMode('local')}
              className={`px-4 sm:px-6 py-2 sm:py-3 rounded-lg font-semibold text-sm sm:text-base transition-all duration-200 shadow-md ${
                gameMode === 'local'
                  ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white transform scale-105'
                  : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
              }`}
            >
              👥 Chơi 2 người
            </button>
            <button
              onClick={() => setGameMode('ai')}
              className={`px-4 sm:px-6 py-2 sm:py-3 rounded-lg font-semibold text-sm sm:text-base transition-all duration-200 shadow-md ${
                gameMode === 'ai'
                  ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white transform scale-105'
                  : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
              }`}
            >
              🤖 Chơi với AI
            </button>
          </div>
          <div className="flex flex-col lg:flex-row gap-4 sm:gap-8">
            <div className="lg:w-80 space-y-4 sm:space-y-6 order-2 lg:order-1">
              <div className="bg-gradient-to-br from-gray-50 to-gray-100 p-4 sm:p-6 rounded-xl border border-gray-200 shadow-lg">
                <h3 className="font-bold text-base sm:text-lg mb-3 text-gray-800">📊 Thông tin ván cờ</h3>
                <div className="space-y-2 text-sm sm:text-base">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Quân bắt được:</span>
                    <span className="font-medium">⚫ {captures.black} | ⚪ {captures.white}</span>
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
                  <TimerDisplay
                    expiryTimestamp={timerExpiry}
                    onExpire={() => {
                      handlePass();
                      alert(`${currentPlayer === Stone.BLACK ? 'Đen' : 'Trắng'} hết thời gian và pass.`);
                    }}
                    isActive={isTimerActive && gameStatus === 'playing'}
                    color={currentPlayer === Stone.BLACK ? 'black' : 'white'}
                  />
                </div>
              </div>
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 sm:p-6 rounded-xl border border-blue-200 shadow-lg">
                <h3 className="font-bold text-base sm:text-lg mb-3 text-gray-800">⚙️ Cài đặt</h3>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm sm:text-base font-medium mb-1 text-gray-700">Kích thước bàn cờ:</label>
                    <select
                      value={settings.boardSize}
                      onChange={(e) => setSettings(prev => ({ ...prev, boardSize: Number(e.target.value) }))}
                      className="w-full p-2 text-sm sm:text-base border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                    >
                      {BOARD_SIZES.map(size => (
                        <option key={size} value={size}>{size}x{size}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm sm:text-base font-medium mb-1 text-gray-700">Thời gian mỗi nước (giây):</label>
                    <input
                      type="number"
                      value={settings.timePerMove}
                      onChange={(e) => setSettings(prev => ({ ...prev, timePerMove: Number(e.target.value) }))}
                      min="10"
                      max="300"
                      className="w-full p-2 text-sm sm:text-base border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                    />
                  </div>
                  {gameMode === 'ai' && (
                    <>
                      <div>
                        <label className="block text-sm sm:text-base font-medium mb-1 text-gray-700">Bạn chơi quân:</label>
                        <div className="flex space-x-2">
                          <button
                            onClick={() => setSettings(prev => ({ ...prev, humanColor: 'black' }))}
                            className={`flex-1 py-2 px-3 text-sm sm:text-base rounded-lg font-medium transition-all duration-200 ${
                              settings.humanColor === 'black'
                                ? 'bg-gradient-to-r from-gray-800 to-black text-white shadow-md'
                                : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                            }`}
                          >
                            ⚫ Đen
                          </button>
                          <button
                            onClick={() => setSettings(prev => ({ ...prev, humanColor: 'white' }))}
                            className={`flex-1 py-2 px-3 text-sm sm:text-base rounded-lg font-medium transition-all duration-200 ${
                              settings.humanColor === 'white'
                                ? 'bg-gradient-to-r from-gray-100 to-white text-black border-2 border-gray-800 shadow-md'
                                : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                            }`}
                          >
                            ⚪ Trắng
                          </button>
                        </div>
                      </div>
                      <div>
                        <label className="block text-sm sm:text-base font-medium mb-1 text-gray-700">Độ khó AI:</label>
                        <select
                          value={settings.difficulty}
                          onChange={(e) => setSettings(prev => ({ ...prev, difficulty: e.target.value as Difficulty }))}
                          className="w-full p-2 text-sm sm:text-base border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
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
              <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 sm:p-6 rounded-xl border border-green-200 shadow-lg">
                <h3 className="font-bold text-base sm:text-lg mb-3 text-gray-800">📜 Lịch sử nước đi</h3>
                <div className="max-h-48 overflow-y-auto scrollbar-thin scrollbar-thumb-green-400 scrollbar-track-green-100">
                  {moveHistory.slice(-10).map((move, index) => (
                    <div
                      key={index}
                      className="flex justify-between text-sm py-1 border-b border-green-200 last:border-0 hover:bg-green-50 transition-colors duration-150"
                    >
                      <span className="font-medium text-gray-600">{moveHistory.length - 9 + index}.</span>
                      <span className={`font-medium ${move.player === Stone.BLACK ? 'text-black' : 'text-gray-500'}`}>
                        {move.player === Stone.BLACK ? '⚫' : '⚪'}
                      </span>
                      <span className="font-mono text-xs sm:text-sm">
                        {move.isPass ? 'Pass' : `${String.fromCharCode(65 + move.position.x)}${settings.boardSize - move.position.y}`}
                      </span>
                      {move.captures > 0 && (
                        <span className="text-red-600 font-bold">+{move.captures}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
              <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 sm:p-6 rounded-xl border border-purple-200 shadow-lg">
                <h3 className="font-bold text-base sm:text-lg mb-3 text-gray-800">💾 Lưu/Tải</h3>
                <div className="space-y-3">
                  <button
                    onClick={saveGame}
                    className="w-full py-2 px-3 bg-gradient-to-r from-purple-600 to-purple-700 text-white text-sm sm:text-base rounded-lg hover:from-purple-700 hover:to-purple-800 transition-all duration-200 shadow-md transform hover:scale-105"
                  >
                    💾 Lưu ván cờ (SGF)
                  </button>
                  <label className="block">
                    <span className="sr-only">Tải ván cờ</span>
                    <input
                      type="file"
                      accept=".sgf"
                      onChange={loadGame}
                      className="block w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-purple-100 file:text-purple-700 hover:file:bg-purple-200 transition-all"
                    />
                  </label>
                </div>
              </div>
            </div>
            <div className="flex-1 order-1 lg:order-2">
              <div className="flex justify-center mb-4 sm:mb-6">{renderBoard}</div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3 mb-4 sm:mb-6">
                <button
                  onClick={handlePass}
                  disabled={gameStatus !== 'playing'}
                  className="px-3 sm:px-6 py-2 sm:py-3 text-sm sm:text-base bg-gradient-to-r from-yellow-500 to-yellow-600 text-white rounded-lg hover:from-yellow-600 hover:to-yellow-700 disabled:from-gray-300 disabled:to-gray-400 shadow-md transition-all duration-200 transform hover:scale-105"
                >
                  Pass (P)
                </button>
                <button
                  onClick={handleUndo}
                  disabled={moveHistory.length === 0}
                  className="px-3 sm:px-6 py-2 sm:py-3 text-sm sm:text-base bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 disabled:from-gray-300 disabled:to-gray-400 shadow-md transition-all duration-200 transform hover:scale-105"
                >
                  ↶ Hoàn tác (Ctrl+Z)
                </button>
                <button
                  onClick={initializeGame}
                  className="px-3 sm:px-6 py-2 sm:py-3 text-sm sm:text-base bg-gradient-to-r from-red-500 to-red-600 text-white rounded-lg hover:from-red-600 hover:to-red-700 shadow-md transition-all duration-200 transform hover:scale-105"
                >
                  🔄 Chơi lại
                </button>
                <button
                  onClick={() => {
                    const score = calculateScore();
                    setGameScore(score);
                    setShowScore(true);
                    setIsTimerActive(false);
                  }}
                  className="px-3 sm:px-6 py-2 sm:py-3 text-sm sm:text-base bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg hover:from-green-600 hover:to-green-700 shadow-md transition-all duration-200 transform hover:scale-105"
                >
                  📊 Tính điểm
                </button>
              </div>
            </div>
          </div>
        </div>
        {showTutorial && (
          <TutorialModal
            isOpen={showTutorial}
            onClose={() => setShowTutorial(false)}
          />
        )}
        {showScore && gameScore && (
          <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50 backdrop-blur-sm p-4 transition-all duration-300">
            <div className="bg-gradient-to-br from-white to-gray-100 p-6 sm:p-8 rounded-2xl max-w-md w-full shadow-2xl transform transition-all duration-300 scale-100">
              <h2 className="text-xl sm:text-2xl font-bold mb-4 text-center text-gray-800">🏆 Kết quả ván cờ</h2>
              <div className="space-y-3 text-sm sm:text-base">
                <div className="flex justify-between p-2 bg-gray-50 rounded-lg">
                  <span>Vùng đất Đen:</span>
                  <span className="font-medium">{gameScore.blackTerritory}</span>
                </div>
                <div className="flex justify-between p-2 bg-gray-50 rounded-lg">
                  <span>Vùng đất Trắng:</span>
                  <span className="font-medium">{gameScore.whiteTerritory}</span>
                </div>
                <div className="flex justify-between p-2 bg-gray-50 rounded-lg">
                  <span>Quân bắt Đen:</span>
                  <span className="font-medium">{captures.black}</span>
                </div>
                <div className="flex justify-between p-2 bg-gray-50 rounded-lg">
                  <span>Quân bắt Trắng:</span>
                  <span className="font-medium">{captures.white}</span>
                </div>
                <div className="flex justify-between p-2 bg-gray-50 rounded-lg">
                  <span>Komi:</span>
                  <span className="font-medium">{gameScore.komi}</span>
                </div>
                <hr className="my-3 border-gray-300" />
                <div className="flex justify-between p-2 bg-gray-800 text-white rounded-lg">
                  <span className="font-bold">Điểm Đen:</span>
                  <span className="font-bold">{gameScore.blackScore.toFixed(1)}</span>
                </div>
                <div className="flex justify-between p-2 bg-gray-100 rounded-lg">
                  <span className="font-bold">Điểm Trắng:</span>
                  <span className="font-bold">{gameScore.whiteScore.toFixed(1)}</span>
                </div>
                <div
                  className={`text-center font-bold text-lg sm:text-xl p-3 rounded-lg ${
                    gameScore.winner === 'draw'
                      ? 'bg-gradient-to-r from-yellow-400 to-yellow-500 text-white'
                      : gameScore.winner === 'black'
                      ? 'bg-gradient-to-r from-gray-800 to-black text-white'
                      : 'bg-gradient-to-r from-gray-100 to-white text-black border-2 border-gray-800'
                  }`}
                >
                  {gameScore.winner === 'draw' ? '🤝 HÒA!' : `🎉 ${gameScore.winner === 'black' ? 'ĐEN' : 'TRẮNG'} THẮNG!`}
                </div>
              </div>
              <div className="flex gap-3 mt-6">
                <button
                  onClick={() => setShowScore(false)}
                  className="flex-1 py-2 sm:py-3 bg-gradient-to-r from-gray-500 to-gray-600 text-white rounded-lg hover:from-gray-600 hover:to-gray-700 transition-all duration-200 shadow-md transform hover:scale-105 text-sm sm:text-base"
                >
                  Đóng
                </button>
                <button
                  onClick={() => {
                    setShowScore(false);
                    initializeGame();
                  }}
                  className="flex-1 py-2 sm:py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all duration-200 shadow-md transform hover:scale-105 text-sm sm:text-base"
                >
                  Ván mới
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default GoGame;
