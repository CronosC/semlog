TEST="tests.test_load_and_transform_ocel"

ts=$(date +%Y%m%dT%H%M%S)
python -m unittest -v "$TEST" 2>&1 | tee "tests/log.log"