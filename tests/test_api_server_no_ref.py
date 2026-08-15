import unittest

from api_server import ModelSpec, TTSRequest, _resolve_reference_inputs


class NoReferenceRequestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ModelSpec(
            id="kohaku004",
            checkpoint="checkpoint.safetensors",
            ref_latent="kohaku-reference.pt",
        )

    def test_model_id_reference_is_used_by_default(self) -> None:
        self.assertEqual(
            _resolve_reference_inputs(self.spec, TTSRequest(text="こんにちは")),
            (None, "kohaku-reference.pt"),
        )

    def test_no_ref_disables_model_id_reference(self) -> None:
        request = TTSRequest(
            text="こんにちは",
            caption="成人女性の低めで落ち着いた声。",
            no_ref=True,
            cfg_scale_caption=3,
            cfg_scale_speaker=0,
        )
        self.assertEqual(_resolve_reference_inputs(self.spec, request), (None, None))
        self.assertTrue(request.no_ref)
        self.assertEqual(request.cfg_scale_caption, 3)
        self.assertEqual(request.cfg_scale_speaker, 0)

    def test_no_ref_wins_over_explicit_reference_fields(self) -> None:
        request = TTSRequest(
            text="こんにちは",
            no_ref=True,
            reference_latent="request-reference.pt",
        )
        self.assertEqual(_resolve_reference_inputs(self.spec, request), (None, None))


if __name__ == "__main__":
    unittest.main()
