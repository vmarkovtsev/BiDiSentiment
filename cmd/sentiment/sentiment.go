package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"

	"gopkg.in/vmarkovtsev/BiDiSentiment.v1"
)

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	batchSize := sentiment.GetBatchSize()
	batch := []string{}
	session, err := sentiment.OpenSession()
	if err != nil {
		log.Fatalf("Creating Tensorflow session: %v", err)
	}
	defer session.Close()
	evaluate := func() {
		if len(batch) == 0 {
			return
		}
		result, err := sentiment.Evaluate(batch, session)
		if err != nil {
			log.Fatalf("Evaluating %d texts: %v", len(batch), err)
		}
		for _, s := range result {
			fmt.Println(s)
		}
		batch = batch[:0]
	}

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
    batch = append(batch, line)
		if len(batch) >= batchSize {
			evaluate()
		}
	}
	evaluate()
}
