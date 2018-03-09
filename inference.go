package sentiment

import (
	"log"
	"regexp"
	"os"

	"gopkg.in/neurosnap/sentences.v1"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"gopkg.in/vmarkovtsev/BiDiSentiment.v1/assets"
)

type model struct {
	graph *tf.Graph
	input1 tf.Output
	input2 tf.Output
	output tf.Output
	batchSize int
	sequenceLength int
}

var (
	instance = loadModel()
	sentenceSplitter = loadSentenceSplitter()
	whitespace = regexp.MustCompile("\\s+")
)

func loadModel() *model {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "3")
	modelBytes, err := assets.Asset("model.pb")
	if err != nil {
		log.Fatalf("failed to load model.pb from the assets: %v", err)
	}
	graph := tf.NewGraph()
	err = graph.Import(modelBytes, "")
	if err != nil {
		log.Fatalf("importing the model: %v", err)
	}
	input1 := graph.Operation("input_1").Output(0)
	input2 := graph.Operation("input_2").Output(0)
	output := graph.Operation("output").Output(0)
	inputShape, err := input1.Shape().ToSlice()
	if err != nil {
		log.Fatalf("Getting the input shape: %v", err)
	}
	batchSize := int(inputShape[0])
	sequenceLength := int(inputShape[1])
	return &model {
		graph: graph,
		input1: input1,
		input2: input2,
		output: output,
		batchSize: batchSize,
		sequenceLength: sequenceLength,
	}
}

func loadSentenceSplitter() *sentences.DefaultSentenceTokenizer {
	sentenceBytes, err := assets.Asset("english.json")
	if err != nil {
		log.Fatalf("failed to load english.json from the assets: %v", err)
	}

	training, err := sentences.LoadTraining(sentenceBytes)
	if err != nil {
		log.Fatalf("failed to load the training data to split sentences: %v", err)
	}
	return sentences.NewSentenceTokenizer(training)
}

func Evaluate(texts []string, session *tf.Session) ([]float32, error) {
	return EvaluateWithProgress(texts, session, func(int, int){})
}

func EvaluateWithProgress(texts []string, session *tf.Session,
		onBatchProcessed func(int, int)) ([]float32, error) {
	// make each subtext span over less than instance.sequenceLength bytes
	splittedTexts := splitTexts(texts)
	batch1 := make([][]uint8, instance.batchSize)
	batch2 := make([][]uint8, instance.batchSize)
	for i := range batch1 {
		batch1[i] = make([]uint8, instance.sequenceLength)
		batch2[i] = make([]uint8, instance.sequenceLength)
	}
	pos := 0
	size := 0
	for _, group := range splittedTexts {
		size += len(group)
	}
	probs := make([]float32, 0, len(texts) + instance.batchSize)
	evaluate := func() error {
		input1, err := tf.NewTensor(batch1)
		if err != nil {
			return err
		}
		input2, err := tf.NewTensor(batch2)
		if err != nil {
			return err
		}
		result, err := session.Run(map[tf.Output]*tf.Tensor{
			instance.input1: input1, instance.input2: input2,
		}, []tf.Output{instance.output}, nil)
		if err != nil {
			return err
		}
		onBatchProcessed(pos, size)
		rawProbs := result[0].Value().([][]float32)
		for _, vec := range rawProbs {
			probs = append(probs, vec[0] / (vec[0] + vec[1]))
		}
		return nil
	}
	for _, group := range splittedTexts {
		for _, text := range group {
			for i, c := range []uint8(text) {
				batch1[pos][instance.sequenceLength-len(text)+i] = c
				batch2[pos][len(text)-i-1] = c
			}
			pos++
			if pos == instance.batchSize {
				err := evaluate()
				if err != nil {
					return nil, err
				}
				pos = 0
			}
		}
	}
	if pos > 0 {
		err := evaluate()
		if err != nil {
			return nil, err
		}
	}
	result := make([]float32, len(texts))
	pos = 0
	for i, group := range splittedTexts {
		accum := float32(0)
		for range group {
			accum += probs[pos]
			pos++
		}
		result[i] = accum / float32(len(group))
	}
	return result, nil
}

func OpenSession() (*tf.Session, error) {
	return tf.NewSession(instance.graph, &tf.SessionOptions{})
}

func GetBatchSize() int {
	return instance.batchSize
}

func GetSequenceLength() int {
	return instance.sequenceLength
}

func splitTexts(texts []string) [][]string {
	splittedTexts := make([][]string, len(texts))
	for i, text := range texts {
		if len(text) <= instance.sequenceLength {
			splittedTexts[i] = []string{text}
		} else {
			sentences := sentenceSplitter.Tokenize(text)
			splittedTexts[i] = make([]string, 0, len(sentences))
			for _, sentence := range sentences {
				if len(sentence.Text) <= instance.sequenceLength {
					splittedTexts[i] = append(splittedTexts[i], sentence.Text)
				} else {
					// TODO(vmarkovtsev): split sentence into chunks
					splitPoints := whitespace.FindAllStringIndex(sentence.Text, -1)
					startPos := 0
					for j, splitPoint := range splitPoints {
						if splitPoint[0] - startPos > instance.sequenceLength {
							if i == 0 || splitPoints[j-1][1] == startPos {
								if startPos == 0 {
									// the best we can do
									splittedTexts[i] = append(splittedTexts[i], sentence.Text[:instance.sequenceLength])
								}
								break
							}
							splittedTexts[i] = append(splittedTexts[i], sentence.Text[startPos:splitPoints[j-1][0]])
							startPos = splitPoints[j-1][1]
						}
					}
				}
			}
		}
	}
	return splittedTexts
}