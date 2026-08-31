export class AsyncEventSubscription<T> implements AsyncIterableIterator<T> {
  private readonly queued: Array<T> = [];
  private readonly waiting: Array<{
    resolve: (result: IteratorResult<T>) => void;
    reject: (error: Error) => void;
  }> = [];
  private ended = false;
  private failure: Error | undefined;

  constructor(private readonly detach: () => void) {}

  push(value: T): void {
    if (this.ended) return;
    const waiter = this.waiting.shift();
    if (waiter === undefined) {
      this.queued.push(value);
    } else {
      waiter.resolve({ done: false, value });
    }
  }

  end(error?: Error): void {
    if (this.ended) return;
    this.ended = true;
    this.failure = error;
    for (const waiter of this.waiting.splice(0)) {
      if (error === undefined) waiter.resolve({ done: true, value: undefined });
      else waiter.reject(error);
    }
  }

  next(): Promise<IteratorResult<T>> {
    const value = this.queued.shift();
    if (value !== undefined) return Promise.resolve({ done: false, value });
    if (this.failure !== undefined) return Promise.reject(this.failure);
    if (this.ended) return Promise.resolve({ done: true, value: undefined });
    return new Promise((resolve, reject) => this.waiting.push({ resolve, reject }));
  }

  return(): Promise<IteratorResult<T>> {
    this.detach();
    this.end();
    return Promise.resolve({ done: true, value: undefined });
  }

  [Symbol.asyncIterator](): AsyncIterableIterator<T> {
    return this;
  }
}
