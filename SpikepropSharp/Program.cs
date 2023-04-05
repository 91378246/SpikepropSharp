using SpikepropSharp.Utility;

const int TRIALS = 10;
const int EPOCHS = 1000;
const int TEST_RUNS = 100;
const double MAX_TIME = 40;
const double TIMESTEP = 0.1;
const double LEARNING_RATE = 1e-2;

Random rnd = new(1);
// XorHelper.RunTest(rnd, TRIALS, EPOCHS, TEST_RUNS, MAX_TIME, TIMESTEP, LEARNING_RATE);
EcgHelper.RunTest(rnd, trials: 1, epochs: 10, testRuns: 100, maxTime: 30, TIMESTEP, LEARNING_RATE);
