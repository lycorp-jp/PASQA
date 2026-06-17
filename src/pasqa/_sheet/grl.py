#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Gradient Reversal Layer utilities."""

import torch


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg().mul(ctx.lambda_), None


def grad_reverse(x, lambda_=1.0):
    """Apply gradient reversal with scaling."""
    return GradReverse.apply(x, lambda_)
