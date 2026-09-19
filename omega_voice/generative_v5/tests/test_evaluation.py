from omega_voice.causal_v4.evaluate import normalized,intelligibility_tokens,distance

def test_orthography_does_not_create_an_intelligibility_failure_or_fake_alignment():
    assert intelligibility_tokens('All right. I will check.')==intelligibility_tokens('Alright, I will check.')
    assert normalized('All right.')!=normalized('Alright.')

def test_real_lexical_and_negation_changes_remain_errors():
    assert distance(intelligibility_tokens('All right, I will check.'),intelligibility_tokens('Alright, I will not check.'))==1
    assert distance(intelligibility_tokens('The key is here.'),intelligibility_tokens('The key was here.'))==1
